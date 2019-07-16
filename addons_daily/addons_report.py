import click
import os
from .utils.helpers import (
    load_data_s3,
    load_raw_pings,
    get_spark,
    dataframe_joiner,
    get_sc,
    load_keyed_hist,
)
from .utils.telemetry_data import *

# from .utils.amo_data import *
from .utils.bq_data import *
from .utils.raw_pings import *
from pyspark.sql import SparkSession, functions as F

DEFAULT_TZ = "UTC"
DATASET_VERSION = 1
OUTPATH = "s3://telemetry-parquet/addons_daily/v{}/".format(DATASET_VERSION)
CORE_FIELDS = [
    "submission_date_s3",
    "client_id",
    "os",
    "country",
    "subsession_length",
    "scalar_parent_browser_engagement_tab_open_event_count",
    "places_bookmarks_count",
    "scalar_parent_browser_engagement_total_uri_count",
    "devtools_toolbox_opened_count",
    "active_ticks",
    "histogram_parent_tracking_protection_enabled",
    "histogram_parent_webext_background_page_load_ms",
]


def expand_addons(main_summary):
    addons_and_users = main_summary.select(CORE_FIELDS +
                                           [F.explode("active_addons")])
    addons_expanded = addons_and_users.select(CORE_FIELDS + ["col.*"])
    return addons_expanded


def agg_addons_report(
    spark,
    date,
    main_summary,
    search_clients_daily,
    events,
    raw_pings,
    **kwargs
):
    """
    This function will create the addons dataset
    """

    addons_expanded = expand_addons(main_summary)
    addons_expanded_day = addons_expanded.filter(
        "submission_date_s3 = '{}'".format(date)
    ).cache()

    main_summary_day = main_summary.filter(
        "submission_date_s3 = '{}'".format(date)
    ).cache()

    # telemetry metrics
    addon_names = get_top_addon_names(addons_expanded_day)
    user_demo_metrics = get_user_demo_metrics(addons_expanded_day)
    engagement_metrics = get_engagement_metrics(addons_expanded_day,
                                                main_summary_day)
    browser_metrics = get_browser_metrics(addons_expanded_day)
    search_metrics = get_search_metrics(search_clients_daily,
                                        addons_expanded_day)
    top_ten_coinstalls = get_top_10_coinstalls(addons_expanded_day)
    # needs to process 30 days in past, use unfiltered dataframes
    trend_metrics = get_trend_metrics(addons_expanded, date)
    event_metrics = install_flow_events(events)
    is_system = get_is_system(addons_expanded)

    keyed_hists = load_keyed_hist(raw_pings)
    # raw pings metrics
    storage_get = get_storage_local_get_time(keyed_hists)
    storage_set = get_storage_local_set_time(keyed_hists)
    startup_time = get_startup_time(keyed_hists)
    bkg_load_time = get_bkgd_load_time(keyed_hists)
    ba_popup_lt = get_ba_popup_load_time(keyed_hists)
    pa_popup_lt = get_pa_popup_load_time(keyed_hists)
    cs_injection_time = get_cs_injection_time(keyed_hists)

    agg = dataframe_joiner(
        [
            addon_names,
            user_demo_metrics,
            engagement_metrics,
            browser_metrics,
            trend_metrics,
            search_metrics,
            event_metrics,
            is_system,
            storage_get,
            storage_set,
            startup_time,
            top_ten_coinstalls,
            bkg_load_time,
            ba_popup_lt,
            pa_popup_lt,
            cs_injection_time,
        ]
    )

    return agg


@click.command()
@click.option("--date", required=True)
@click.option(
    "--output",
    default=OUTPATH,
    help="Output directory for parquet dataset. Defaults to {}".format(OUTPATH),
)
@click.option("--main_summary_version", default="v4")
@click.option("--search_clients_daily_version", default="v7")
@click.option("--events_version", default="v1")
@click.option("--sample", default=1, help="percent sample as int [1, 100]")
def main(
    date,
    output,
    sample,
    main_summary_version,
    search_clients_daily_version,
    events_version,
):
    if not any(output.startswith(prefix) for prefix in ["file://", "s3://"]):
        raise ValueError(
            "Unsupported data source, "
            "`--output` must start with either `file://` or `s3://`."
        )
    if not output.endswith("/"):
        output = output + "/"

    # path = '' # need to pass in from command line i think
    # path var is a path to the user credentials.json for BQ
    spark = get_spark(DEFAULT_TZ)
    sc = get_sc()
    sc.setLogLevel("INFO")
    fdate = lambda x: x[:4] + "-" + x[4:6] + "-" + x[6:]
    base_date = "to_date('{}')".format(fdate(date))
    # leave main_summary fitlered to last 28 days
    # to get trend metrics
    sample_ids = list(range(0, sample))
    main_summary = (
        load_data_s3(
            spark,
            input_bucket="telemetry-parquet",
            input_prefix="main_summary",
            input_version=main_summary_version,
        )
        .filter(F.col("sample_id").isin(sample_ids))
        .filter("submission_date_s3 >= ({} - INTERVAL 28 DAYS)".format(base_date))
        .filter("normalized_channel = 'release'")
    )

    search_daily = (
        load_data_s3(
            spark,
            input_bucket="telemetry-parquet",
            input_prefix="search_clients_daily",
            input_version=search_clients_daily_version,
        )
        .filter(F.col("sample_id").isin(sample_ids))
        .filter("submission_date_s3 = '{}'".format(date))
        .filter("channel = 'release'")
    )

    events = (
        load_data_s3(
            spark,
            input_bucket="telemetry-parquet",
            input_prefix="events",
            input_version=events_version,
        )
        .filter(F.col("sample_id").isin(sample_ids))
        .filter("submission_date_s3 = '{}'".format(date))
        .filter("normalized_channel = 'release'")
    )

    raw_pings = load_raw_pings(sc, date)

    # bq_d = load_bq_data(datetime.date.today(), path, spark)

    agg_data = agg_addons_report(
        spark, date, main_summary, search_daily, events, raw_pings
    )

    # write to s3
    (
        agg_data.drop("submission_date_s3")
        .repartition(10)
        .write.mode("overwrite")
        .parquet(output + "submission_date_s3={}".format(date))
    )


if __name__ == "__main__":
    main()

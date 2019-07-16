from addons_daily.utils.helpers import *
# from addons_daily.addons_report import expand_addons
import pyspark.sql.functions as F
import pandas as pd
from pyspark.sql import SQLContext, Row
from pyspark.sql.window import Window
from pyspark.sql.types import StringType
from itertools import chain


TOP_COUNTRIES = {
    "US": "United States",
    "DE": "Germany",
    "FR": "France",
    "IN": "India",
    "BR": "Brazil",
    "CN": "China",
    "ID": "Indonesia",
    "RU": "Russia",
    "IT": "Italy",
    "PL": "Poland",
    "GB": "Great Britain",
    "CA": "Canada",
}

TOP_OS = {"Windows_NT", "Darwin", "Linux"}


def source_map(df, extra_filter=""):
    m = F.create_map(
        list(
            chain(
                *(
                    (F.lit(name.split("_")[0]), F.col(name))
                    for name in df.columns
                    if name != "addon_id" and extra_filter in name
                )
            )
        )
    ).alias(extra_filter)
    return m


def bucket_field(field, ref):
    """
    :param field: A string
    :param ref: List of possible buckets
    :return: field if field is one of the possible buckets, or else "Other"
    """
    if field in ref:
        return field
    return "Other"


@F.udf(StringType())
def bucket_os(os):
    """
    :param os: A string
    :return: os if os is one of the TOP_OS, or else "Other"
    """
    return bucket_field(os, TOP_OS)


@F.udf(StringType())
def bucket_country(country):
    """
    :param country: A string
    :return: country if country is one of the TOP_COUNTRIES, or else "Other"
    """
    return bucket_field(country, TOP_COUNTRIES)


def get_user_demo_metrics(addons_expanded):
    """
    :param addons_expanded: addons_expanded dataframe
    :return: aggregated dataframe by addon_id
        with user demographic information including
        - distribution of users by operating system
        - distribution of users by country
    """

    def source_map(df, extra_filter=""):
        m = F.create_map(
            list(
                chain(
                    *(
                        (F.lit(name), F.col(name))
                        for name in df.columns
                        if name != "addon_id"
                    )
                )
            )
        ).alias(extra_filter)
        return m

    client_counts = (
        addons_expanded.groupBy("addon_id")
        .agg(F.countDistinct("client_id").alias("total_clients"))
        .cache()
    )

    os_dist = (
        addons_expanded.withColumn("os", bucket_os("os"))
        .groupBy("addon_id", "os")
        .agg(F.countDistinct("client_id").alias("os_client_count"))
        .join(client_counts, on="addon_id", how="left")
        .withColumn("os_pct",
                    F.col("os_client_count") / F.col("total_clients"))
        .groupby("addon_id")
        .pivot("os")
        .agg(F.first("os_pct").alias("os_pct"))
    )

    os_dist = (
        os_dist
        .na.fill(0)
        .select("addon_id", source_map(os_dist, "os_pct"))
    )

    ct_dist = (
        addons_expanded.withColumn("country", bucket_country("country"))
        .groupBy("addon_id", "country")
        .agg(F.countDistinct("client_id").alias("country_client_count"))
        .join(client_counts, on="addon_id", how="left")
        .withColumn(
            "country_pct",
            F.col("country_client_count") / F.col("total_clients")
        )
        .groupBy("addon_id")
        .pivot("country")
        .agg(F.first("country_pct").alias("country_pct"))
    )

    ct_dist = (
        ct_dist
        .na.fill(0)
        .select("addon_id", source_map(ct_dist, "country_pct"))
    )

    combined_dist = os_dist.join(ct_dist, on="addon_id", how="outer")
    return combined_dist


#####################################################################
# User Engagement Metrics - total hours, active hours, addon disabled
#####################################################################


def get_engagement_metrics(addons_expanded, main_summary):
    """
    :param addons_expanded: addons_expanded
    :param main_summary: main_summary
    :return: dataframe aggregated by addon_id
        with engagement metrics including
        - average total hours spent per user
        - average total active ticks per user
        - number of clients with the addon disabled
    """
    engagement_metrics = (
        addons_expanded.groupBy("addon_id", "client_id", "submission_date_s3")
        .agg(
            F.sum("active_ticks").alias("total_ticks"),
            F.sum("subsession_length").alias("daily_total"),
        )
        .groupBy("addon_id")
        .agg(
            F.mean("total_ticks").alias("avg_time_active_ms"),
            F.mean("daily_total").alias("avg_time_total"),
        )
        .withColumn("active_hours", F.col("avg_time_active_ms") / (12 * 60))
        .drop("avg_time_active_ms")
    )

    disabled_addons = (
        main_summary.where(F.col("disabled_addons_ids").isNotNull())
        .withColumn("addon_id", F.explode("disabled_addons_ids"))
        .groupBy("addon_id")
        .agg((F.countDistinct("client_id") * F.lit(100)).alias("disabled"))
    )

    engagement_metrics = engagement_metrics.join(
        disabled_addons, on="addon_id", how="outer"
    )

    return engagement_metrics


#######################################################################
# Browser Metrics - avg tabs, avg bookmarks, avg devtools opened count,
#   avg URI, and percent with tracking protection enabled
#######################################################################


def get_browser_metrics(addons_expanded):
    """
    :param addons_expanded: addons_expanded
    :return: dataframe aggregated by addon_id
        with browser-related metrics including
            - average number of tabs open
            - average number of bookmarks
            - average devtools opened
            - average URI
            - percent of users with tracking enabled
    """
    browser_metrics = addons_expanded.groupby("addon_id").agg(
        F.avg("scalar_parent_browser_engagement_tab_open_event_count").alias(
            "avg_tabs"
        ),
        F.avg("places_bookmarks_count").alias("avg_bookmarks"),
        F.avg("devtools_toolbox_opened_count")
        .alias("avg_toolbox_opened_count"),
    )

    avg_uri = (
        addons_expanded.groupBy("addon_id", "client_id")
        .agg(
            F.mean("scalar_parent_browser_engagement_total_uri_count")
            .alias("avg_uri")
        )
        .groupBy("addon_id")
        .agg(F.mean("avg_uri").alias("avg_uri"))
    )

    tracking_enabled = (
        addons_expanded.where(
            "histogram_parent_tracking_protection_enabled is not null"
        )
        .select(
            "addon_id",
            "histogram_parent_tracking_protection_enabled.1",
            "histogram_parent_tracking_protection_enabled.0",
        )
        .groupBy("addon_id")
        .agg(F.sum("1"), F.count("0"))
        .withColumn("total", F.col("sum(1)") + F.col("count(0)"))
        .withColumn("pct_w_tracking_prot_enabled",
                    F.col("sum(1)") / F.col("total"))
        .drop("sum(1)", "count(0)", "total")
    )

    browser_metrics = (
        browser_metrics
        .join(avg_uri, on="addon_id", how="outer")
        .join(tracking_enabled, on="addon_id", how="outer")
    )

    return browser_metrics


def get_top_10_coinstalls(addons_expanded_day):
    """
    :param addons_expanded_day: addons_expanded from the last day
    :return: dataframe aggregated by addon_id
        with a list of the top 10 addon ids installed by
        clients with addon_id installed
    """
    def str_map_to_dict(m):
        result = {}
        for i in m:
            k, v = i.split("=")
            result[k] = v
        return result

    def format_row(row):
        return Row(
            addon_id=row.addon_id,
            top_10_coinstalls=str_map_to_dict(row.top_10_coinstalls),
        )

    w = Window().partitionBy("addon_id").orderBy(F.col("count").desc())
    d = (
        addons_expanded_day.join(
            addons_expanded_day.filter("is_system=false").withColumnRenamed(
                "addon_id", "coaddon"
            ),
            on="client_id",
        )
        .groupby("client_id", "addon_id", "coaddon")
        .count()
        .withColumn("rn", (F.row_number().over(w) - F.lit(1)))  # start at 0
        .filter("rn BETWEEN 1 and 10")  # ignore addon where coaddon==addon_id
        .groupby("addon_id")
        .agg(
            F.collect_list(F.concat(F.col("rn"), F.lit("="), "coaddon")).alias(
                "top_10_coinstalls"
            )
        )
        .rdd.map(format_row)
        .toDF()
    )

    return d


###############################
# Trend Metrics - DAU, WAU, MAU
###############################


def get_trend_metrics(addons_expanded, date):
    """
    :param df: addons_expanded
    :return: aggregated dataframe by addon_id
        with trend metrics including
        - daily active users
        - weekly active users
        - monthly active users
    """
    fdate = lambda x: x[:4] + "-" + x[4:6] + "-" + x[6:]
    base_date = "to_date('{}')".format(fdate(date))

    # limit to last 30 days to calculate mau
    addons_expanded = (
        addons_expanded.withColumn("date", F.to_date("submission_date_s3", "yyyyMMdd"))
        .filter("date >= ({} - INTERVAL 28 DAYS)".format(base_date))
        .filter("date <= {}".format(base_date))
    )
    mau = addons_expanded.groupby("addon_id").agg(
        (F.countDistinct("client_id") * F.lit(100)).alias("mau")
    )

    # limit to last 7 days to calculate wau
    addons_expanded_week = addons_expanded.filter(
        "date >= ({} - INTERVAL 7 DAYS)".format(base_date)
    ).filter("date <= {}".format(base_date))

    wau = addons_expanded_week.groupby("addon_id").agg(
        (F.countDistinct("client_id") * F.lit(100)).alias("wau")
    )

    # limit to last 1 day to calculate dau
    addons_expanded_day = addons_expanded_week.filter("date = {}".format(base_date))
    absolute_dau = addons_expanded_day.select("client_id").distinct().count()

    dau = addons_expanded_day.groupby("addon_id").agg(
        (F.countDistinct("client_id") * F.lit(100)).alias("dau")
    )

    dau_pct = (
        dau
        .withColumn("dau_prop", F.col("dau") / absolute_dau)
        .drop(F.col("dau"))
    )

    trend_metrics = (
        mau.join(wau, on="addon_id", how="outer")
        .join(dau, on="addon_id", how="outer")
        .join(dau_pct, on="addon_id", how="outer")
    )

    return trend_metrics


def get_top_addon_names(addons_expanded):
    """
    :param addons_expanded: the addons_expanded df
    :return: A data frame with addon id and name in descending order by
    number of clients
    """
    w = Window().partitionBy("addon_id").orderBy(F.col("n").desc())
    cnts = addons_expanded.groupby("addon_id", "name").agg(
        F.countDistinct("client_id").alias("n")
    )
    addon_names = (
        cnts.withColumn("rn", F.row_number().over(w))
        .where(F.col("rn") == 1)
        .select("addon_id", "name")
    )
    return addon_names


def install_flow_events(events):
    """
    :param events: The events dataframe
    :return: Dataframe aggregated by addon_id with
        - number of installs
        - average download time
        - number of uninstalls
    """
    def source_map(df, alias, extra_filter=""):
        m = F.create_map(
            list(
                chain(
                    *(
                        (F.lit(name.split("_")[0]), F.col(name))
                        for name in df.columns
                        if name != "addon_id" and extra_filter in name
                    )
                )
            )
        ).alias(alias)
        return m

    install_flow_events = (
        events.select(
            [
                "client_id",
                "submission_date_s3",
                "event_map_values.method",
                "event_method",
                "event_string_value",
                "event_map_values.source",
                "event_map_values.download_time",
                "event_map_values.addon_id",
            ]
        )
        .filter("event_category = 'addonsManager'")
        .filter("event_object = 'extension'")
        .filter(
            """
        (event_method = 'install'
        and event_map_values.step = 'download_completed')
        or
        (event_method = 'uninstall')
        """
        )
        .withColumn(
            "addon_id",
            F.when(
                F.col("addon_id").isNull(),
                F.col("event_string_value")
            )
            .otherwise(F.col("addon_id")),
        )  # uninstalls populate addon_id in a different place
        .drop("event_string_value")
        .groupby("addon_id", "event_method", "source")
        .agg(
            F.avg("download_time").alias("avg_download_time"),
            F.countDistinct("client_id").alias("n_distinct_users"),
        )
    )

    installs = (
        install_flow_events.filter("event_method = 'install'")
        .groupBy("addon_id")
        .pivot("source")
        .agg(
            (F.sum("n_distinct_users") * F.lit(100)),
            F.avg("avg_download_time")
        )
    )
    uninstalls = (
        install_flow_events.filter("event_method = 'uninstall'")
        .groupBy("addon_id")
        .pivot("source")
        .agg((F.sum("n_distinct_users") * F.lit(100)))
    )

    agg = (
        installs.na.fill(0)
        .select(
            "addon_id",
            source_map(installs, "installs", "n_distinct_users"),
            source_map(installs, "download_times", "avg_download_time"),
        )
        .join(
            uninstalls.na.fill(0).select(
                "addon_id", source_map(uninstalls, "uninstalls")
            ),
            on="addon_id",
            how="full",
        )
    )

    return agg
    # number_installs = (
    #     install_flow_events
    #     .where(install_flow_events.event_method == "install")
    #     .groupby("addon_id")
    #     .agg(F.sum("n_distinct_users").alias("installs"))
    # )

    # number_uninstalls = (
    #     install_flow_events
    #     .where(install_flow_events.event_method == "uninstall")
    #     .groupby("addon_id")
    #     .agg(F.sum("n_distinct_users").alias("uninstalls"))
    # )

    # install_flow_events_df = number_installs.join(
    #     number_uninstalls, "addon_id", how="full"
    # )

    # return install_flow_events_df


def get_search_metrics(search_daily_df, addons_expanded):
    """
    :param search_daily_df: search clients daily dataframe
    :param addons_expanded: addons_expanded dataframe
    :return: aggregated dataframe by addon_id
        with metris from the search clients daily dataset
        - average sap searches by search engine
        - average tagged sap searches by search engine
        - average organic searches by search engine
        - total number of searches with ads by search engine
        - total number of ad clicks by search engine
    """
    search_daily_df = bucket_engine(search_daily_df)

    user_addon = addons_expanded.select("client_id", "addon_id")
    user_addon_search = user_addon.join(search_daily_df, "client_id")

    df = (
        user_addon_search.groupBy("addon_id")
        .pivot("engine")
        .agg(
            F.sum("sap").alias("sap_searches"),
            F.sum("tagged_sap").alias("tagged_sap_searches"),
            F.sum("organic").alias("organic_searches"),
            F.sum("search_with_ads").alias("search_with_ads"),
            F.sum("ad_click").alias("ad_click"),
        )
    )

    df_mapped = df.na.fill(0).select(
        "addon_id",
        source_map(df, "search_with_ads"),
        source_map(df, "ad_click"),
        source_map(df, "organic_searches"),
        source_map(df, "sap_searches"),
        source_map(df, "tagged_sap_searches"),
    )

    return df_mapped


def get_is_system(addons_expanded):
    """
    :param addons_expanded: addons_expanded
    :return: dataframe containing addon_id
        and boolean variable is_system indicating
        whether the addon is a system addon
    """
    is_system = (
        addons_expanded.select(["addon_id", "is_system"])
        .groupby("addon_id")
        .agg(F.first("is_system").alias("is_system"))
    )

    return is_system

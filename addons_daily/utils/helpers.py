import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark.sql import SparkSession
from pyspark import SparkContext

from moztelemetry import Dataset
import datetime
import os


make_map = F.udf(
    lambda x, y: dict(zip(x, y)),
    MapType(StringType(), DoubleType())
)


# taken from Fx_Usage_Report
def get_dest(
    output_bucket,
    output_prefix,
    output_version,
    date=None,
    sample_id=None
):
    """
    Stiches together an s3 destination.
    :param output_bucket: s3 output_bucket
    :param output_prefix: s3 output_prefix (within output_bucket)
    :param output_version: dataset output_version
    :retrn str ->
    s3://output_bucket/output_prefix/output_version/submissin_date_s3=[date]/sample_id=[sid]
    """
    suffix = ""
    if date is not None:
        suffix += "/submission_date_s3={}".format(date)
    if sample_id is not None:
        suffix += "/sample_id={}".format(sample_id)
    full_dest = (
        "s3://"
        + "/".join([output_bucket, output_prefix, output_version])
        + suffix
        + "/"
    )
    return full_dest


# taken from Fx_Usage_Report
def load_data_s3(spark, input_bucket, input_prefix, input_version):
    """
    Loads main_summary from the bucket constructed from
    input_bucket, input_prefix, input_version
    :param spark: SparkSession object
    :param input_bucket: s3 bucket (telemetry-parquet)
    :param input_prefix: s3 prefix (main_summary)
    :param input_version: dataset version (v4)
    :return SparkDF
    """
    dest = get_dest(input_bucket, input_prefix, input_version)
    return spark.read.option("mergeSchema", True).parquet(dest)


def load_raw_pings(sc, date):
    """
    Function to load raw pings data
    :param sc: a spark context
    :return a spark dataframe of raw pings
    """
    raw_pings = (
        Dataset.from_source("telemetry")
        .where(docType="main")
        .where(appUpdateChannel="release")
        .where(submissionDate=date)
        .records(sc, sample=0.01)
    )
    return raw_pings


def load_keyed_hist(rp):
    """
    :param rp: dataframe of raw_pings returned from load_raw_pings()
    :return: just the keyed histograms
    """
    return rp.map(lambda x: x["payload"]["keyedHistograms"]).cache()


# def load_bq_data(credential_path, project="ga-mozilla-org-prod-001"):
#     """
#     Function to load data from big-query
#     :param credential_path: path to the JSON file of your credentials for BQ
#     :param project: the string project path, only pass if different than the
#        standard project above
#     :return: the data from bigquery in form of list of dictionary per row
#     """
#     client = bigquery.Client(project=project)
#     os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credential_path
#     query = (
#         "SELECT * "
#         "FROM `ga-mozilla-org-prod-001.67693596.ga_sessions_20190219` "
#         "LIMIT 100"
#     )
#     query_job = client.query(query, location="US")
#     return [dict(row.items()) for row in query_job]


def histogram_mean(values):
    """
    Returns the mean of values in a histogram.
    This mean relies on the sum *post*-quantization, which amounts to a
    left-hand-rule discrete integral of the histogram. It is therefore
    likely to be an underestimate of the true mean.
    """
    if values is None:
        return None
    numerator = 0
    denominator = 0
    for k, v in values.items():
        numerator += int(k) * v
        denominator += v
    if denominator == 0:
        return None
    return numerator / float(denominator)


def get_hist_avg(hist, just_keyed_hist):
    """
    :param hist: name of histogram of interest
    :param just_keyed_hist: pyspark dataframe of keyed histograms
    :return: returns a pyspark dataframe aggregated in the following form:
    addon_id : mean(hist)
    """
    hist_data = (
        just_keyed_hist.filter(lambda x: hist in x.keys())
        .map(lambda x: x[hist])
        .flatMap(lambda x: [(i, histogram_mean(x[i]["values"]))
                 for i in x.keys()])
    )

    agg_schema = StructType(
        [
            StructField("addon_id", StringType(), True),
            StructField("avg_" + hist.lower(), FloatType(), True),
        ]
    )

    return hist_data.toDF(schema=agg_schema)


def dataframe_joiner(dfs):
    """
    Given a list of dataframes, join them all on "addon_id",
    and return the joined dataframe
    :param dfs: list of pyspark aggregated dfs
    :return: one joined df of all the dataframes in dfs
    """
    left = dfs[0]
    for right in dfs[1:]:
        left = left.join(right, on="addon_id", how="left")
    return left


def take_top_ten(l):
    """
    Given a list of values, find the top 10
    :param l: a list
    :return: the top 10 items of l
    """
    if len(l) < 10:
        return sorted(l, key=lambda i: -list(i.values())[0])
    else:
        return sorted(l, key=lambda i: -list(i.values())[0])[0:10]


def get_spark(tz="UTC"):
    """
    Creates Spark session and sets the time zone
    :param tz: The time zone
    :return: A Spark session
    """
    spark = SparkSession.builder.appName("usage_report").getOrCreate()

    spark.conf.set("spark.sql.session.timeZone", tz)

    return spark


def get_sc():
    """
    Creates a Spark context
    :return: A Spark context
    """
    sc = SparkContext.getOrCreate()
    return sc


def list_expander(lis):
    """
    :param lis: A list
    :return: A list of lists. Each item is a list containing an item
    from the original list and a list of all of the other items in the list
    """
    list_of_lists = []
    for item in lis:
        list_of_lists.append([item, [i for i in lis if i != item]])
    return list_of_lists


def bucket_engine(df):
    """
    :param df: a dataframe with 'engine' column
    :return: dataframe where the engine column is bucketed into
        'google', 'duckduckgo', 'bing', or 'other'
    """
    eng = F.lower(F.col("engine"))
    return df.withColumn(
        "engine",
        F.when(eng.like("google%"), "google")
        .when(eng.like("ddg%"), "duckduckgo")
        .when(eng.like("duckduckgo%"), "duckduckgo")
        .when(eng.like("bing%"), "bing")
        .otherwise("other"),
    )


def str_to_list(word):
    """
    :param word: a string containing a list of words
    :return: a list of words
    """
    if word[0] == "[":
        word = word[1:]
    if word[-1] == "]":
        word = word[:-1]
    return [x.strip() for x in word.split(",")]


def is_same(df, expected_df, verbose=False):
    """
    Determines whether two dataframes are the same, regardless of
    order of rows and columns
    :param df: A dataframe
    :param expected_df: A second dataframe to compare with
    :return: True if the dataframes are the same, false otherwise
    """
    cols = sorted(df.columns)
    intersection = df.select(*cols).intersect(expected_df.select(*cols))
    df_len, expected_len, actual_len = (
        df.count(),
        expected_df.count(),
        intersection.count(),
    )

    if verbose:
        print("\nInput Dataframe\n")
        print(df.select(*cols).collect())
        print("\nExpected Dataframe\n")
        print(expected_df.collect())

    assert df_len == expected_len
    assert actual_len == expected_len, "Missing {} Rows".format(
        expected_len - actual_len
    )

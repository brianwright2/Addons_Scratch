from .helpers import *
import pyspark.sql.functions as F
import pandas as pd
import re
from pyspark.sql import SQLContext


def agg_hist_keyed_by_addon(df, hist):
    """
    :param df: Raw pings dataframe
    :param hist: name of histogram of interest
    :return: Dataframe aggregated by addon id, with the averaged histogram
    """
    new_hist = re.sub("_webext", "", hist.lower().split("by_addonid")[0])
    agg = get_hist_avg(hist, df)
    agg_by_addon = agg.groupBy("addon_id").agg(
        F.mean("avg_" + hist.lower()).alias("avg_" + new_hist)
    )
    return agg_by_addon


def get_storage_local_get_time(df):
    """
    :param df: Raw pings dataframe
    :return: Dataframe aggregated by addon id with the average storage
        local get time
    """
    return agg_hist_keyed_by_addon(
        df,
        "WEBEXT_STORAGE_LOCAL_GET_MS_BY_ADDONID"
    )


def get_storage_local_set_time(df):
    """
    :param df: Raw pings dataframe
    :return: Dataframe aggregated by addon id with the average storage
        local set time
    """
    return agg_hist_keyed_by_addon(
        df,
        "WEBEXT_STORAGE_LOCAL_SET_MS_BY_ADDONID"
    )


def get_startup_time(df):
    """
    :param df: Raw pings dataframe
    :return: Dataframe aggregated by addon id with the average startup time
    """
    return agg_hist_keyed_by_addon(
        df,
        "WEBEXT_EXTENSION_STARTUP_MS_BY_ADDONID"
    )


def get_bkgd_load_time(df):
    """
    :param df: Raw pings dataframe
    :return: Dataframe aggregated by addon id with the average background
        page load time
    """
    return agg_hist_keyed_by_addon(
        df,
        "WEBEXT_BACKGROUND_PAGE_LOAD_MS_BY_ADDONID"
    )


def get_ba_popup_load_time(df):
    """
    :param df: Raw pings dataframe
    :return: Dataframe aggregated by addon id with the average browser action
        popup load time
    """
    return agg_hist_keyed_by_addon(
        df,
        "WEBEXT_BROWSERACTION_POPUP_OPEN_MS_BY_ADDONID"
    )


def get_pa_popup_load_time(df):
    """
    :param df: Raw pings dataframe
    :return: Dataframe aggregated by addon id with the average page action
        popup load time
    """
    return agg_hist_keyed_by_addon(
        df,
        "WEBEXT_PAGEACTION_POPUP_OPEN_MS_BY_ADDONID"
    )


def get_cs_injection_time(df):
    """
    :param df: Raw pings dataframe
    :return: Dataframe aggregated by addon id with the average content script
        injection time
    """
    return agg_hist_keyed_by_addon(
        df,
        "WEBEXT_CONTENT_SCRIPT_INJECTION_MS_BY_ADDONID"
    )

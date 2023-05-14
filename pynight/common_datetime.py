import datetime
import pytz


##
def datetime_dir_name(timezone='Asia/Tehran'):
    current_datetime = datetime.datetime.now(pytz.timezone(timezone))
    date_string = current_datetime.strftime("%Y_%b_%d/%H_%M")
    return date_string


##

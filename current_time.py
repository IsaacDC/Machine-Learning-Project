import datetime
import pytz

# Only fetches time in EST for now
def get_current_time():
    est = pytz.timezone('US/Eastern')
    utc_time = datetime.datetime.now()
    est_time = utc_time.astimezone(est)

    hours = str(est_time.strftime('%I'))
    minutes = str(est_time.minute)
    am_pm = est_time.strftime('%p')

    # If minutes < 10, no 0 is added infront of single digit
    # This fixes that issue
    if len(minutes) == 1:
        minutes = "0" + minutes

    return hours + ":" + minutes + am_pm

# print(get_current_time())
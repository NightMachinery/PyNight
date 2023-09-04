import csv
from io import StringIO


##
def dict_to_csv(data, header_p=True):
    """
    Convert a dictionary into a CSV string.

    Args:
        data (dict): The dictionary to be converted into CSV.
        header_p (bool, optional): Whether to include fieldnames as the first row in the CSV. Default is True.

    Returns:
        str: A CSV formatted string.
    """
    # Create a StringIO buffer to write the CSV data
    csv_buffer = StringIO()

    fieldnames = data.keys()

    csv_writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)

    if header_p:
        csv_writer.writeheader()

    # Write the data as a single row
    csv_writer.writerow(data)

    # Get the CSV content as a string
    csv_string = csv_buffer.getvalue()

    csv_buffer.close()

    return csv_string

##

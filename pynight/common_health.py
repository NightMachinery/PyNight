##
def bmi(*, weight, height):
    """
    Calculate the Body Mass Index (BMI) given weight in kilograms and height in centimeters.
    BMI is defined as a person's weight, in kilograms, divided by the square of the person's height, in meters.

    Args:
    weight (float): Weight in kilograms.
    height (float): Height in centimeters.

    Returns:
    float: The calculated BMI.
    """
    height_in_meters = height / 100  # Convert height from cm to meters
    bmi_value = weight / (height_in_meters**2)  # Calculate BMI
    return bmi_value


##

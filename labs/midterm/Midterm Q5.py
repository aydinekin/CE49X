# Test data for Question 5
test_dates = ["15/06/2024", "28/02/2023", "01/01/2020"]

# Your solution here
def parse_construction_date(date_str):
    day, month, year = map(int, date_str.split("/")) #string is splitted
    is_leap_year = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0) #taken with condition
    return (day, month, year), is_leap_year

# Test your solution
for date in test_dates:
    result = parse_construction_date(date)
    print(f"Date: {date}")
    print(f"Parsed: {result[0]}")
    print(f"Is leap year: {result[1]}\n")
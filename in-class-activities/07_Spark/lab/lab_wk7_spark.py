from pyspark.sql import *

spark = SparkSession.builder \
    .appName("Political Speech Analysis") \
    .getOrCreate()


data = [
        ("we enacted tax credits of $800 per person per year to reduce healthcare costs for millions of working families", "Politician A"),
        ("To demonstrate the necessity of education in our government, I shall not attempt to derive any proofs from the history of other Republics", "Politician A"),
        ("So, we here agree that access to healthcare should be a right and not just a privilege of those who can afford it", "Politician B"),
        ("We have also taken on the crisis — the healthcare crisis of maternal mortality", "Politician A"),
        ("This summer, I traveled to Chicago, where I spoke about a fundamentally changing American economy", "Politician A"),
        ("Today, I'm here to tell you that, in Education, too, the journey is the destination.", "Politician B"),
    ]

# The sentences of speeches are from:
# https://www.whitehouse.gov/briefing-room/speeches-remarks/2024/04/03/remarks-by-president-biden-on-lowering-healthcare-costs-for-americans-2/#:~:text=I%20exact%20tax%20credits%20%E2%80%94%20we,it%20for%20%E2%80%94%20through%20this%20year.
# https://www.c-span.org/video/?530449-1/presidential-remarks-economy
# https://www.whitehouse.gov/briefing-room/speeches-remarks/2022/05/10/remarks-by-president-biden-on-the-economy-5/
# https://www.whitehouse.gov/briefing-room/statements-releases/2022/06/24/fact-sheet-president-bidens-maternal-health-blueprint-delivers-for-women-mothers-and-families/
# https://www.ed.gov/news/speeches/remarks-us-secretary-education-miguel-cardona-raise-bar-lead-world

# Keywords to filter relevant policy topics
keywords =  ['healthcare', 'economy']

# TODO
# Filter: filter the speeches to include only those that contain any of the keywords
# Map: map each filtered entry to a key-value pair where the key is the politician and the value is 1
# Reduce: sum up all values associated with the same key (politician)
speeches_rdd = spark.sparkContext.parallelize(data) \
                 .filter() \
                 .map() \
                 .reduceByKey()  \
                 .collect()

print(speeches_rdd)

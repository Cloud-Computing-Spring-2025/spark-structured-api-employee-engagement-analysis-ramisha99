# task1_identify_departments_high_satisfaction.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, round as spark_round

def initialize_spark(app_name="Task1_Identify_Departments"):
    """
    Initialize and return a SparkSession.
    """
    spark = SparkSession.builder \
        .appName(app_name) \
        .getOrCreate()
    return spark

def load_data(spark, file_path):
    """
    Load the employee data from a CSV file into a Spark DataFrame.

    Parameters:
        spark (SparkSession): The SparkSession object.
        file_path (str): Path to the employee_data.csv file.

    Returns:
        DataFrame: Spark DataFrame containing employee data.
    """
    schema = "EmployeeID INT, Department STRING, JobTitle STRING, SatisfactionRating INT, EngagementLevel STRING, ReportsConcerns BOOLEAN, ProvidedSuggestions BOOLEAN"
    
    df = spark.read.csv(file_path, header=True, schema=schema)
    return df

def identify_departments_high_satisfaction(df):
    """
    Identify departments with more than 3% of employees having a Satisfaction Rating > 4 and Engagement Level 'High'.
    """
    # 1. Filter employees with SatisfactionRating > 4 and EngagementLevel == 'High'.
    high_satisfaction = df.filter((col("SatisfactionRating") > 4) & (col("EngagementLevel") == "High"))
    
    # 2. Count total employees and those with high satisfaction in each department
    department_counts = df.groupBy("Department").agg(count("*").alias("total_employees"))
    high_satisfaction_counts = high_satisfaction.groupBy("Department").agg(count("*").alias("high_satisfaction_employees"))
    
    # 3. Join the two counts to calculate the percentage
    result = department_counts.join(high_satisfaction_counts, on="Department", how="left")
    
    # 4. Calculate the percentage and filter departments with more than 3% high satisfaction
    result = result.withColumn("satisfaction_percentage", 
                               (col("high_satisfaction_employees") / col("total_employees")) * 100)
    
    result = result.filter(col("satisfaction_percentage") > 3).select("Department", 
                                                                     spark_round(col("satisfaction_percentage"), 2).alias("Percentage"))
    return result



def write_output(result_df, output_path):
    """
    Write the result DataFrame to a CSV file.

    Parameters:
        result_df (DataFrame): Spark DataFrame containing the result.
        output_path (str): Path to save the output CSV file.

    Returns:
        None
    """
    result_df.coalesce(1).write.csv(output_path, header=True, mode='overwrite')

def main():
    """
    Main function to execute Task 1.
    """
    # Initialize Spark
    spark = initialize_spark()
    
    # Define file paths
    input_file = "/workspaces/spark-structured-api-employee-engagement-analysis-ramisha99/input/employee_data.csv"
    output_file = "/workspaces/spark-structured-api-employee-engagement-analysis-ramisha99/outputs/task1/departments_high_satisfaction.csv"
    
    # Load data
    df = load_data(spark, input_file)
    
    # Perform Task 1
    result_df = identify_departments_high_satisfaction(df)
    
    # Write the result to CSV
    write_output(result_df, output_file)
    
    # Stop Spark Session
    spark.stop()

if __name__ == "__main__":
    main()

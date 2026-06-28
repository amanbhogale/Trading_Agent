#!/bin/bash

# Navigate to the project directory
cd /home/zombie/Documents/Trading_Agent

# Unset SPARK_HOME as it causes issues with local PySpark execution
unset SPARK_HOME

# Run the ETL script and log output to /tmp
/home/zombie/Documents/Agentic/bin/python src/trading_system/pyspark_etl.py >> /tmp/pyspark_etl.log 2>&1

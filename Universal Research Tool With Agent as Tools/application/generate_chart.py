import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
import os

def forecast_monthly_deposite_or_expense(json_data: dict, chart_type: str = "deposit", forecast_months: int = 6) -> tuple:
    """
    Generates a monthly deposite/expense forecast plot and a table of forecasted values
    for the specified number of months from a given JSON financial transaction data.

    Args:
        json_data (dict): A dictionary containing transaction data in the format:
                          {"transactions": [{"ID": ..., "Type": "CR/DR", "Amount": ..., "Date": ..., ...}]}
        chart_type (str): Type of chart to generate - "deposit" or "expenses"
        forecast_months (int): Number of months to forecast (default: 6)

    Returns:
        tuple: (image_path, forecast_table) - Path to the saved image and forecast table
    """
    warnings.filterwarnings("ignore")

    # Hardcoded output directory
    output_dir = "static/charts"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Convert to DataFrame
    df = pd.DataFrame(json_data['transactions'])

    # Filter transactions based on chart type
    if chart_type.lower() == "deposit":
        filtered_df = df[df['Type'] == 'CR'].copy()
        title_prefix = "Monthly Deposite"
    else:  # expenses
        filtered_df = df[df['Type'] == 'DR'].copy()
        title_prefix = "Monthly Expenses"

    # Convert 'Date' to datetime objects
    filtered_df['Date'] = pd.to_datetime(filtered_df['Date'])

    # Set 'Date' as index
    filtered_df.set_index('Date', inplace=True)

    # Resample to monthly frequency and sum the 'Amount'
    monthly_data = filtered_df['Amount'].resample('MS').sum()

    # Convert to DataFrame for easier plotting and manipulation
    monthly_df = monthly_data.reset_index()
    monthly_df.rename(columns={'Date': 'ds', 'Amount': 'y'}, inplace=True)
    monthly_df['ds'] = monthly_df['ds'].dt.to_period('M')
    monthly_df = monthly_df.set_index('ds')

    # Define SARIMAX parameters (simplified for small datasets)
    order = (1, 0, 0)
    seasonal_order = (0, 0, 0, 0)

    # Fit SARIMAX Model
    model = SARIMAX(monthly_df['y'], order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit(disp=False)

    # Forecast for the specified number of months
    forecast_result = model_fit.get_forecast(steps=forecast_months)
    forecast_mean = forecast_result.predicted_mean
    forecast_conf_int = forecast_result.conf_int()

    # Create dates for the forecast period
    last_date_period = monthly_df.index.max()
    forecast_period_index = pd.period_range(start=last_date_period + 1, periods=forecast_months, freq='M')

    # Prepare forecast data for plotting
    forecast_df = pd.DataFrame({
        'yhat': forecast_mean.values,
        'yhat_lower': forecast_conf_int['lower y'].values,
        'yhat_upper': forecast_conf_int['upper y'].values
    }, index=forecast_period_index)

    # --- Monthly Forecast Plot ---
    plt.figure(figsize=(14, 7))
    
    # Plot historical data
    plt.plot(monthly_df.index.to_timestamp(), monthly_df['y'], 
             label=f'Historical {title_prefix}', marker='o', color='blue')
    
    # Plot forecast data
    plt.plot(forecast_df.index.to_timestamp(), forecast_df['yhat'], 
             label=f'Forecasted {title_prefix} ({forecast_months} months)', 
             color='red', linestyle='--', marker='x')
    
    # Add confidence interval
    plt.fill_between(forecast_df.index.to_timestamp(), 
                     forecast_df['yhat_lower'], forecast_df['yhat_upper'], 
                     color='pink', alpha=0.3, 
                     label=f'95% Confidence Interval ({forecast_months} months)')
    
    # Add vertical line to separate historical and forecast data
    last_historical_date = monthly_df.index.to_timestamp()[-1]
    plt.axvline(x=last_historical_date, color='gray', linestyle='--', alpha=0.5,
                label='Forecast Start')
    
    plt.title(f'{title_prefix} Forecast (SARIMAX Model) - Next {forecast_months} Months', fontsize=18)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Total Amount', fontsize=14)
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save plot to file
    image_path = os.path.join(output_dir, f'monthly_forecast.png')
    plt.savefig(image_path, format='png', bbox_inches='tight')
    plt.close()

    # --- Forecasted Values Table ---
    forecast_output = forecast_df[['yhat', 'yhat_lower', 'yhat_upper']].copy()
    forecast_output.index = forecast_output.index.strftime('%Y-%m')
    forecast_table = forecast_output.to_markdown(numalign="left", stralign="left")

    return image_path, forecast_table

# Example usage:
if __name__ == "__main__":
    # Provided JSON data
    data = {"transactions": [ {"ID": 1, "UserID": "U001", "Type": "CR", "Amount": 2000.0, "Date": "2024-08-05", "ToPerson": "Payroll", "AccountNumber": "1111222233334444", "Description": "Monthly salary"}, {"ID": 2, "UserID": "U001", "Type": "DR", "Amount": 120.0, "Date": "2024-08-06", "ToPerson": "Netflix", "AccountNumber": "1111222233334444", "Description": "Streaming subscription"}, {"ID": 3, "UserID": "U001", "Type": "DR", "Amount": 85.0, "Date": "2024-08-10", "ToPerson": "Amazon", "AccountNumber": "1111222233334444", "Description": "Online shopping"}, {"ID": 4, "UserID": "U001", "Type": "CR", "Amount": 150.0, "Date": "2024-08-15", "ToPerson": "Freelance", "AccountNumber": "1111222233334444", "Description": "Side project payment"}, {"ID": 5, "UserID": "U001", "Type": "DR", "Amount": 45.0, "Date": "2024-08-20", "ToPerson": "Spotify", "AccountNumber": "1111222233334444", "Description": "Music subscription"}, {"ID": 6, "UserID": "U001", "Type": "CR", "Amount": 2000.0, "Date": "2024-09-05", "ToPerson": "Payroll", "AccountNumber": "1111222233334444", "Description": "Monthly salary"}, {"ID": 7, "UserID": "U001", "Type": "DR", "Amount": 120.0, "Date": "2024-09-06", "ToPerson": "Netflix", "AccountNumber": "1111222233334444", "Description": "Streaming subscription"}, {"ID": 8, "UserID": "U001", "Type": "CR", "Amount": 200.0, "Date": "2024-09-12", "ToPerson": "Freelance", "AccountNumber": "1111222233334444", "Description": "Consulting work"}, {"ID": 9, "UserID": "U001", "Type": "DR", "Amount": 95.0, "Date": "2024-09-18", "ToPerson": "Walmart", "AccountNumber": "1111222233334444", "Description": "Groceries"}, {"ID": 10, "UserID": "U001", "Type": "CR", "Amount": 2000.0, "Date": "2024-10-05", "ToPerson": "Payroll", "AccountNumber": "1111222233334444", "Description": "Monthly salary"}, {"ID": 11, "UserID": "U001", "Type": "DR", "Amount": 120.0, "Date": "2024-10-06", "ToPerson": "Netflix", "AccountNumber": "1111222233334444", "Description": "Streaming subscription"}, {"ID": 12, "UserID": "U001", "Type": "CR", "Amount": 180.0, "Date": "2024-10-15", "ToPerson": "Freelance", "AccountNumber": "1111222233334444", "Description": "Project completion"}, {"ID": 13, "UserID": "U001", "Type": "DR", "Amount": 75.0, "Date": "2024-10-22", "ToPerson": "Target", "AccountNumber": "1111222233334444", "Description": "Household items"}, {"ID": 14, "UserID": "U001", "Type": "CR", "Amount": 2000.0, "Date": "2024-11-05", "ToPerson": "Payroll", "AccountNumber": "1111222233334444", "Description": "Monthly salary"}, {"ID": 15, "UserID": "U001", "Type": "DR", "Amount": 120.0, "Date": "2024-11-06", "ToPerson": "Netflix", "AccountNumber": "1111222233334444", "Description": "Streaming subscription"}, {"ID": 16, "UserID": "U001", "Type": "CR", "Amount": 250.0, "Date": "2024-11-14", "ToPerson": "Freelance", "AccountNumber": "1111222233334444", "Description": "Extra project"}, {"ID": 17, "UserID": "U001", "Type": "DR", "Amount": 110.0, "Date": "2024-11-25", "ToPerson": "Best Buy", "AccountNumber": "1111222233334444", "Description": "Electronics"}, {"ID": 18, "UserID": "U001", "Type": "CR", "Amount": 2000.0, "Date": "2024-12-05", "ToPerson": "Payroll", "AccountNumber": "1111222233334444", "Description": "Monthly salary"}, {"ID": 19, "UserID": "U001", "Type": "DR", "Amount": 120.0, "Date": "2024-12-06", "ToPerson": "Netflix", "AccountNumber": "1111222233334444", "Description": "Streaming subscription"}, {"ID": 20, "UserID": "U001", "Type": "CR", "Amount": 300.0, "Date": "2024-12-10", "ToPerson": "Freelance", "AccountNumber": "1111222233334444", "Description": "Year-end project"}, {"ID": 21, "UserID": "U001", "Type": "DR", "Amount": 150.0, "Date": "2024-12-20", "ToPerson": "Amazon", "AccountNumber": "1111222233334444", "Description": "Holiday shopping"}, {"ID": 22, "UserID": "U001", "Type": "CR", "Amount": 2000.0, "Date": "2025-01-05", "ToPerson": "Payroll", "AccountNumber": "1111222233334444", "Description": "Monthly salary"}, {"ID": 23, "UserID": "U001", "Type": "DR", "Amount": 120.0, "Date": "2025-01-06", "ToPerson": "Netflix", "AccountNumber": "1111222233334444", "Description": "Streaming subscription"}, {"ID": 24, "UserID": "U001", "Type": "CR", "Amount": 220.0, "Date": "2025-01-15", "ToPerson": "Freelance", "AccountNumber": "1111222233334444", "Description": "New year project"}, {"ID": 25, "UserID": "U001", "Type": "DR", "Amount": 90.0, "Date": "2025-01-25", "ToPerson": "Walmart", "AccountNumber": "1111222233334444", "Description": "Monthly groceries"}] }
    
    # Example usage for both deposit and expenses
    # image_path_deposit, forecast_table_deposit = forecast_monthly_deposite_or_expense(data, chart_type="deposit", forecast_months=6)
    # print(f"Deposit Chart saved to: {image_path_deposit}")
    # print("\n--- Forecasted Deposits for the Next 6 Months ---")
    # print(forecast_table_deposit)

    image_path_expenses, forecast_table_expenses = forecast_monthly_deposite_or_expense(data, chart_type="expenses", forecast_months=2)
    print(f"\nExpenses Chart saved to: {image_path_expenses}")
    print("\n--- Forecasted Expenses for the Next 6 Months ---")
    print(forecast_table_expenses)
import os
import requests
import re
import streamlit as st
import pandas as pd
import plotly.express as px
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.api import VAR
from io import BytesIO
from pyxlsb import open_workbook as open_xlsb

# Core analysis functions (we previously defined these)
def load_data(file_path):
    df = pd.read_excel(file_path)
    df = df.set_index('Date')
    df.index = df.index.astype('period[Q-DEC]')
    df.index = df.index.astype(str)
    return df

def plot_time_series(data, var, title):
    fig = px.line(data, x=data.index, y=var, title=title)
    return fig

def check_stationarity(data, variables):
    def adf_test(series):
        result = adfuller(series, autolag='AIC')
        return result[1]  # Return the p-value
    
    adf_pvalues = {var: adfuller(data[var], autolag='AIC')[1] for var in variables}
    return adf_pvalues

def apply_differencing(data, variables):
    differenced_data = data.copy()
    for var in variables:
        differenced_data[var] = data[var].diff()
    return differenced_data.dropna()

def granger_causality(data, target_variable, predictor_variables, maxlag):
    test = 'ssr_chi2test'
    causality_results = {}
    for var in predictor_variables:
        results = grangercausalitytests(data[[var, target_variable]], maxlag=maxlag, verbose=False)
        p_values = [round(results[i+1][0][test][1], 4) for i in range(maxlag)]
        causality_results[var] = p_values
    return causality_results
    
def highlight_small_values(val):
    """
    Highlight values less than 0.05 in green.
    """
    color = 'green' if val < 0.05 else 'red'
    return f'color: {color}'
    
def remove_alphabets(value):
    if isinstance(value, str):
        return re.sub(r'[a-zA-Z\s,]', '', value)
    return value

# Streamlit app
st.title("Time Series Analysis for Exchange Rate Determinants")
st.write("Upload your Excel dataset for time series analysis:")

uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

if uploaded_file:
    data = pd.read_excel(uploaded_file)
    country_name = pd.read_excel(uploaded_file, sheet_name = "Quarterly").columns[0].split(":")[0]
    st.write(f'Country: {country_name}')
    df_bop = pd.read_excel(uploaded_file, header = 4, sheet_name = "Quarterly")
    df_bop = df_bop.set_index('Unnamed: 0').T
    df_bop.index = df_bop.index.astype('period[Q-DEC]')
    df_bop = df_bop.iloc[:,:-6]
    
    # Check for the existence of the file
    if not os.path.exists("BOP_Exchange.xlsx"):
        with st.spinner("Downloading Exchange Rate Data..."):
    
            # Specify the URL to download the file from
            url1 = "https://github.com/safi842/Exchange-Rate-Prediction/releases/download/v1/BOP_Exchange.xlsx"
            url2 = "https://github.com/safi842/Exchange-Rate-Prediction/releases/download/v2/Abhiraj.-.Profits.xlsx"
            url3 = "https://github.com/safi842/Exchange-Rate-Prediction/releases/download/v2/Abhiraj.-.PE.Ratios.xlsx"
            url4 = "https://github.com/safi842/Exchange-Rate-Prediction/releases/download/v2/Abhiraj.-.Commodity.Prices.xlsx"
    
            # Download the file and save it in the current directory
            response1 = requests.get(url1)
            with open("BOP_Exchange.xlsx", "wb") as f:
                f.write(response1.content)
            response2 = requests.get(url2)
            with open('Abhiraj - Profits.xlsx', "wb") as f:
                f.write(response2.content)
            response3 = requests.get(url3)
            with open("Abhiraj - PE Ratios.xlsx", "wb") as f:
                f.write(response3.content)
            response4 = requests.get(url4)
            with open("Abhiraj - Commodity Prices.xlsx", "wb") as f:
                f.write(response4.content)
            st.write("Files downloaded successfully.")

    country_loc = list(pd.read_excel('BOP_Exchange.xlsx',sheet_name = 'Nominal', header = 3).columns).index(country_name)
    df_ex = pd.read_excel('BOP_Exchange.xlsx',sheet_name = 'Nominal', header = 4)
    df_ex.index = df_ex.set_index('Unnamed: 0').index.to_period('Q')
    df_ex = df_ex[~df_ex.index.duplicated(keep='last')]
    df_ex = df_ex[[df_ex.columns[country_loc]]]
    df_ex.columns = ['Exchange Rate']
   
    df = df_bop.merge(df_ex, left_index=True, right_index=True, how='left')
    df = df.drop(columns=df.columns[df.eq("...").any()])
    df = df.applymap(remove_alphabets).apply(pd.to_numeric)
    df_profits = pd.read_excel('Abhiraj - Profits.xlsx') 
    df_pe = pd.read_excel('Abhiraj - PE Ratios.xlsx')
    df_compr = pd.read_excel('Abhiraj - Commodity Prices.xlsx')
    
    # PROFIT
    df_profits['Date'] = pd.to_datetime(df_profits['Date'])

    # Set 'Date' as the index
    df_profits.set_index('Date', inplace=True)

    # Resample the data quarterly and compute the mean for each quarter
    quarterly_df_profits = df_profits.resample('Q').mean()

    quarterly_df_profits.index = quarterly_df_profits.index.to_period('Q')
    quarterly_df_profits = quarterly_df_profits.diff(periods =4).dropna()
    quarterly_df_profits = quarterly_df_profits[[country_name]]
    quarterly_df_profits.columns = ['Profits diff']
    
    
    # PE
    df_pe['Date'] = pd.to_datetime(df_pe['Date'])

    # Set 'Date' as the index
    df_pe.set_index('Date', inplace=True)

    # Resample the data quarterly and compute the mean for each quarter
    quarterly_df_pe = df_pe.resample('Q').mean()

    quarterly_df_pe.index = quarterly_df_pe.index.to_period('Q')
    quarterly_df_pe[quarterly_df_pe < 0] = 0
    quarterly_df_pe = quarterly_df_pe[[country_name]]
    quarterly_df_pe.columns = ['PE ratio']
    
    # COMMODITY PRICES
    df_compr['Date'] = pd.to_datetime(df_compr['Date'])

    # Set 'Date' as the index
    df_compr.set_index('Date', inplace=True)

    # Resample the data quarterly and compute the mean for each quarter
    quarterly_df_compr = df_compr.resample('Q').mean()

    quarterly_df_compr.index = quarterly_df_compr.index.to_period('Q')
    quarterly_df_compr = quarterly_df_compr.diff(periods = 4).dropna()
    quarterly_df_compr = quarterly_df_compr[[country_name]]
    quarterly_df_compr.columns = ['Commodity Prices Diff']
    
    
    df = df.merge(quarterly_df_profits, left_index=True, right_index=True, how='left')
    df = df.merge(quarterly_df_pe, left_index=True, right_index=True, how='left')
    df = df.merge(quarterly_df_compr, left_index=True, right_index=True, how='left')
    # Rename 'Credit' and 'Debit' columns by adding prefixes based on the preceding column
    new_columns = []
    prefix = None
    for col in df.columns:
        if 'Credit' in col:
            prefix = new_columns[-1]  # Take the last added column name as the prefix
            new_columns.append(f"{prefix}_Credit")
        elif 'Debit' in col:
            new_columns.append(f"{prefix}_Debit")
        else:
            new_columns.append(col)
    # Update the DataFrame with the new column names
    df.columns = new_columns
    df = df.rename_axis('Date').reset_index()
    df['Date'] = df['Date'].astype('datetime64[ns]')
    df = df.set_index('Date')
    df = df.dropna('index')
    data = df

    #data = load_data(uploaded_file)
    all_variables = list(data.columns)
    selected_variables = st.multiselect("Choose variables for analysis:", all_variables, default=['Current account', 'Goods', 'Services', 'Exchange Rate'])
    
    data =  data[selected_variables]
    
    st.write('Data Preview:')
    st.write(data.head())
    
    def to_excel(df):
        output = BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        df.to_excel(writer, sheet_name='Sheet1')
        workbook = writer.book
        worksheet = writer.sheets['Sheet1']
        format1 = workbook.add_format({'num_format': '0.00'}) 
        worksheet.set_column('A:A', None, format1)  
        writer.save()
        processed_data = output.getvalue()
        return processed_data
    df_xlsx = to_excel(data[selected_variables])
    st.download_button(label='Download Data',
                                data=df_xlsx ,
                                file_name= 'Full_Data.xlsx')
    
    st.subheader("Original Time Series Plots")
    st.plotly_chart(plot_time_series(data, selected_variables, 'Original Time Series'))
    
    

    st.subheader("Augmented Dickey-Fuller (ADF) Test")
    st.write("The Augmented Dickey-Fuller (ADF) test is used to determine the presence of unit root in the series, which helps in understanding if the series is stationary or not. The null hypothesis of the ADF test is that the time series possesses a unit root and is non-stationary. If the p-value is less than a critical size (e.g., 0.05), then the null hypothesis is rejected and the time series is considered stationary.")
    
    adf_results = check_stationarity(data, selected_variables)
    diff_option = st.checkbox("Would you like to perform differenciation to make variables stationary?")
    if diff_option:
        diff_variables = st.multiselect("Select variables to difference", ['Current account', 'Goods', 'Services','PE ratio'] , default = ['Current account', 'Goods', 'Services'])
        differenced_data = apply_differencing(data, diff_variables)
        adf_results = check_stationarity(differenced_data, selected_variables)
    else:
        differenced_data = data
        adf_results = check_stationarity(differenced_data, selected_variables)
   
    adf_table = pd.DataFrame(adf_results.items(), columns=["Variable", "p-value"])
    adf_table["Stationarity"] = adf_table["p-value"].apply(lambda x: "Stationary" if x < 0.05 else "Non-stationary")
    st.table(adf_table)


    st.subheader("Granger Causality Test for Exchange Rate")
    st.write("The Granger causality test is used to determine if one time series can predict another time series. It's based on the idea that if variable X Granger-causes variable Y, then past values of X should contain information that helps predict Y. If the p-value for a given lag is less than a critical size (e.g., 0.05), we can conclude that the time series in the row Granger causes the time series in the column for that lag.")
    
    target_variable = "Exchange Rate"
    predictor_variables = [var for var in selected_variables if var != target_variable]
    maxlag = st.slider("Select maxlag for Granger causality test:", 1, 10, 5)
    granger_results = granger_causality(data, target_variable, predictor_variables, maxlag)
    granger_df = pd.DataFrame(granger_results).T
    granger_df.columns = [f"Lag {i+1}" for i in range(maxlag)]
    st.write(granger_df.style.applymap(highlight_small_values))
    #st.table(granger_df)
    
    
    if diff_option:
        st.subheader("Differenced Time Series Plots")
        st.plotly_chart(plot_time_series(differenced_data, selected_variables, 'Differenced Time Series'))
    
    
    st.subheader("Vector Autoregression (VAR) Model")
    st.write("The Vector Autoregression (VAR) model captures the linear interdependencies among multiple time series. All variables in a VAR enter the model in the same way: each variable has an equation, where it is regressed against its own lagged values and the lagged values of all the other variables.")
    lag_order = st.slider("Select the lag order for the VAR model:", 1, 20, 5)
    st.write("Fitting a VAR model to the differenced data with selected lag order.")

    differenced_data = apply_differencing(data, selected_variables)
    var_model = VAR(differenced_data).fit(lag_order)
    with st.expander("VAR Model Summary"):
        st.text(str(var_model.summary()))

    st.subheader("Impulse Response Analysis")
    st.write("Impulse Response Analysis is used to analyze the dynamic effects of shocks to the system over time. It shows the response trajectory of all the variables in the system to a one-time shock to each of the variables. It helps in understanding how shocks to one variable (e.g., a policy change) will impact other variables in subsequent periods.")
    
    st.write("Analyzing the dynamic effects of shocks to the system over time.")
    impulse_responses = var_model.irf(10)  # Get impulse responses for 10 periods

    metrics = []
    response_var = "Exchange Rate"
    response_var_index = selected_variables.index(response_var)
    for j, shock_var in enumerate(selected_variables):
        if shock_var != "Exchange Rate":  # Exclude the case where the shock is also in Exchange Rate
            fig = px.line(x=list(range(11)), y=impulse_responses.orth_irfs[:, selected_variables.index(response_var), selected_variables.index(shock_var)], 
                      title=f"Response of {response_var} to {shock_var}", labels={'x': 'Periods', 'y': 'Response'})
            with st.expander(f"Impulse Response Plots - {shock_var}"):
                st.plotly_chart(fig)
            # Calculating metrics
            responses = [
                impulse_responses.orth_irfs[i][response_var_index][j]
                for i in range(11)
            ]
            max_response = max(responses, key=abs)
            max_response_lag = responses.index(max_response)
            metrics.append((shock_var, max_response, max_response_lag))
        
        # Display the metrics in a table
    df_metrics = pd.DataFrame(metrics, columns=["Shocked Variable", "Max Response", "Lag of Max Response"])
    st.table(df_metrics)
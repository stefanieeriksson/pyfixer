#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

class PandasHelper:
    def __init__(self):
        pass

    # Pandas df shorthand functions #
    @staticmethod
    def df(input_for_dataframe):
        """return pd.DataFrame(x) """
        return pd.DataFrame(input_for_dataframe)

    @staticmethod
    def svd(data1, col):
        """return df.sort_values(x, ascending=False"""
        return data1.sort_values(col, ascending=False)

    @staticmethod
    def sva(data1, col):
        """return df.sort_values(x, ascending=True"""
        return data1.sort_values(col, ascending=True)

    @staticmethod
    def vcdf_all(data1, col):
        """returns df[{column}].value_counts().reset_index() """
        return data1[col].value_counts().reset_index()

    @staticmethod
    def vcdf(data, cols, n, ex_others=False):
        """
        Returns a DataFrame containing the top n value counts for specified columns
        along with an additional group aggregating all other values. Also includes 
        percentage of total counts.

        Parameters:
        - data: Pandas DataFrame containing the data.
        - cols: Column name or list of column names to count unique values.
        - n: Number of top values to display.

        Returns:
        - DataFrame with categories, their counts, percentage of total, and other stats.
        """
        # Perform value counts on multiple columns if necessary
        value_counts = data.value_counts(subset=cols).reset_index(name='count')
        total = value_counts['count'].sum()

        # Sorting and limiting the results
        value_counts.sort_values(by='count', ascending=False, inplace=True)
        if not ex_others and len(value_counts) > n:
            top_n = value_counts.head(n)
            others_count = value_counts.iloc[n:]['count'].sum()
            # Handle single and multiple columns
            if isinstance(cols, list):
                others_row = pd.DataFrame([["All Other Values"] * len(cols) + [others_count]], columns=cols+['count'])
            else:
                others_row = pd.DataFrame({cols: ["All Other Values"], 'count': [others_count]})
            top_n = pd.concat([top_n, others_row], ignore_index=True)
        else:
            top_n = value_counts.head(n)

        # Calculate additional statistics
        # Use .loc to ensure setting values on the actual DataFrame, not a copy
        with pd.option_context('mode.chained_assignment', None):
            top_n.loc[:, 'perc_of_total'] = top_n['count'] / total
            top_n.loc[:, 'cumulative_count'] = top_n['count'].cumsum()
            top_n.loc[:, 'cumulative_perc'] = top_n['perc_of_total'].cumsum()
            top_n.loc[:, 'perc_of_top'] = top_n['count'] / top_n.iloc[0, top_n.columns.get_loc('count')]
            top_n.loc[:, 'rank'] = range(1, len(top_n) + 1)
        
        # Round values after calculations
        top_n.loc[:, 'perc_of_total'] = top_n['perc_of_total'].round(2)
        top_n.loc[:, 'cumulative_perc'] = top_n['cumulative_perc'].round(2)
        top_n.loc[:, 'perc_of_top'] = top_n['perc_of_top'].round(2)

        return top_n         

    @staticmethod
    def vcdf20(data1, col):
        """returns df[{column}].value_counts().reset_index().head(20)"""
        return PandasHelper.vcdf(data1, col, 20)

    @staticmethod
    def vcdf10(data1, col):
        """returns df[{column}].value_counts().reset_index().head(10)"""
        return PandasHelper.vcdf(data1, col, 10)

    @staticmethod
    def vcdf7(data1, col):
        """returns df[{column}].value_counts().reset_index().head(7)"""
        return PandasHelper.vcdf(data1, col, 7)

    @staticmethod
    def vc(df, col):
        """returns df[{column}].value_counts()"""
        return df[col].value_counts()


    @staticmethod
    def perc(df, extended=True, round=2):
        """
        Computes percentiles for numeric and datetime columns in a DataFrame.

        Args:
        df (pd.DataFrame): Input DataFrame from which to calculate percentiles.

        Returns:
        pd.DataFrame: A DataFrame with percentiles as rows (0.1 to 0.9) and relevant columns.
        """
        # Filter DataFrame for numeric and datetime columns
        numeric_and_date_cols = df.select_dtypes(include=[np.number, 'datetime64[ns]'])

        # Define the percentiles
        percentiles = np.linspace(0.1, 0.9, 9)
        
        # Compute percentiles
        perc_df = numeric_and_date_cols.quantile(percentiles, interpolation='lower')
        x = df.describe()

        # Custom rounding function
        def custom_round(x):
            if x.dtype == float:
                return x.round(round)  # Round floats to two decimal places
            elif x.dtype == int:
                return x
            return x

        if extended:
            perc_df = pd.concat([x, perc_df], sort=False)
            # perc_df = perc_df.round(0).apply(lambda x: x.astype(int) if not x.isnull().any() else x)
            perc_df = perc_df.apply(custom_round)
            return perc_df
        else:
            return perc_df


    @staticmethod  
    def aggregate(df, groupby, distinct=None, max=None, min=None, avg=None, sum=None, std=None, median=None, round_agg=True, decimals=2):
        # Start the aggregation dictionary
        agg_dict = {'rows': 'size'}  # Counting rows in each group
        df['rows'] = 1

        # Helper function to add aggregation functions to dictionary
        def add_to_agg_dict(cols, func_name):
            if cols:
                if isinstance(cols, list):
                    for col in cols:
                        agg_dict[col] = func_name
                else:
                    agg_dict[cols] = func_name

        # Add distinct counts
        if distinct:
            if isinstance(distinct, list):
                for col in distinct:
                    df[f'distinct_{col}'] = df[col]
                    agg_dict[f'distinct_{col}'] = pd.Series.nunique
            else:
                df[f'distinct_{distinct}'] = df[distinct]
                agg_dict[f'distinct_{distinct}'] = pd.Series.nunique

        # Add other aggregations
        add_to_agg_dict(max, 'max')
        add_to_agg_dict(min, 'min')
        add_to_agg_dict(sum, 'sum')
        add_to_agg_dict(avg, 'mean')
        add_to_agg_dict(std, 'std')
        add_to_agg_dict(median, 'median')

        # Group by the specified column and aggregate
        grouped_df = df.groupby(groupby, observed=True).agg(agg_dict).reset_index()


        # Apply rounding after the aggregation if necessary
        if round_agg:
            for cols in [avg, std, median]:
                if cols:
                    if isinstance(cols, list):
                        for col in cols:
                            grouped_df[col] = grouped_df[col].round(decimals)

                    else:
                        grouped_df[cols] = grouped_df[cols].round(decimals)
                
            
        # Order by row count descending
        grouped_df = grouped_df.sort_values(by='rows', ascending=False)

        # Clean up the DataFrame by dropping temporary columns
        if distinct:
            if isinstance(distinct, list):
                for col in distinct:
                    df.drop(f'distinct_{col}', axis=1, inplace=True)
            else:
                df.drop(f'distinct_{distinct}', axis=1, inplace=True)

        return grouped_df

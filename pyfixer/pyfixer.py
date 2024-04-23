#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import re
from datetime import date
import matplotlib.pyplot as plt
from pandas.api.types import is_categorical_dtype
import numpy as np

class PyFix:
    def __init__(self):
        pass

    # Pandas df shorthand functions #
    @staticmethod
    def df(input_for_dataframe):
        """return pd.DataFrame(x) """
        return pd.DataFrame(input_for_dataframe)

    @staticmethod
    def dfx(input_for_dataframe):
        """return pd.DataFrame(x) """
        return pd.DataFrame(input_for_dataframe)

    @staticmethod
    def svd(df, col):
        """return df.sort_values(x, ascending=False"""
        return df.sort_values(col, ascending=False)

    @staticmethod
    def sva(df, col):
        """return df.sort_values(x, ascending=True"""
        return df.sort_values(col, ascending=True)

    @staticmethod
    def vcdf_all(df, col):
        """returns df[{column}].value_counts().reset_index() """
        return df[col].value_counts().reset_index()

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
    def vcdf20(df, col):
        """returns df[{column}].value_counts().reset_index().head(20)"""
        return PyFix.vcdf(df, col, 20)

    @staticmethod
    def vcdf10(df, col):
        """returns df[{column}].value_counts().reset_index().head(10)"""
        return PyFix.vcdf(df, col, 10)

    @staticmethod
    def vcdf7(df, col):
        """returns df[{column}].value_counts().reset_index().head(7)"""
        return PyFix.vcdf(df, col, 7)

    @staticmethod
    def vc(df, col):
        """returns df[{column}].value_counts()"""
        return df[col].value_counts()

    @staticmethod
    def snake(name):
        """Convert camelCase to snake_case."""
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

    @staticmethod
    def snake_df(df):
        """Convert DataFrame column names to snake_case."""
        new_columns = {col: PyFix.snake(col) for col in df.columns}
        return df.rename(columns=new_columns)


    # [listcomprehension shorthand functions] #
    @staticmethod
    def not_in(list_a, exclude_these):
        """returns [item for item in list_a if item not in exclude_these]"""
        return [item for item in list_a if item not in exclude_these]

    @staticmethod
    def any_in(in_this_text, find_these_strings):
        """" Returns True if AT LEAST ONE of the strings of the [] are in the text"""
        return any(s in in_this_text for s in find_these_strings)

    @staticmethod
    def all_in(in_this_text, find_these_strings):
        """" Returns True if ALL of the strings of the [] are in the text"""
        return all(s in in_this_text for s in find_these_strings)

    # List [] shorthand functions #
    @staticmethod
    def pl(list_a, text=None):
        """print(len(x))"""
        if not text:
            print(len(list_a))
        else:
            print(text, len(list_a)) 

    @staticmethod
    def pls(list_a, leading_text):
        """#print({leading text}, len(x))"""
        print(leading_text, len(list_a)) 

    @staticmethod
    def rdup(list_a):
        """"Removes duplicates from a list"""
        list_a = set(list_a)
        return list(list_a)
    
    # General shorthand functions #
    @staticmethod
    def to_int(value, default=0):
        """try return int(value) except return default """
        try:
            if value:
                if value.strip().lower().endswith('k'):
                    # Remove the 'k' and convert the remaining number to float, then multiply by 1000
                    number = float(value[:-1]) * 1000
                    return int(number)
                else:
                    return int(value)
        except:
            pass

        return default

    @staticmethod
    def find_between(text, first, stop_at):
        splits = text.split(first)
        try:
            if len(splits) > 1:
                split2 = splits[1].split(stop_at)
                return split2[0].strip()
        except:
            pass

    @staticmethod               
    def to_date (date_value, default=None):
        try:
            if date_value:
                return date.fromisoformat(date_value)
        except:
            pass
        
        return default

    @staticmethod 
    def unsnake(snake_str, keep_casing=False):
        """
        Converts a snake case string to a title-like format.
        
        Parameters:
            snake_str (str): The snake case string to convert.
            keep_casing (bool): If True, retains the original casing, otherwise capitalizes each word.
        
        Returns:
            str: The converted string in title-like format.
        """
        if snake_str:
            if keep_casing:
                # Split the string on underscores and join with a space
                return ' '.join(snake_str.split('_'))
            else:
                # Split the string, capitalize each word, and join with a space
                return ' '.join(word.capitalize() for word in snake_str.split('_'))

    @staticmethod
    def hist(df, col_name, n, ex_others=False, sort_dim=False, max_label_length=20, return_df=False, vertical=0):
        """
        Plots a horizontal bar chart of the top n values of the specified column.
        
        Parameters:
        - df: Pandas DataFrame containing the data to plot.
        - col_name: Name of the column to count unique values and plot.
        - n: Number of top values to display.

        The chart will be sorted in descending order of counts, and bars will be colored uniquely per category.
        """
        plotdf = PyFix.vcdf(df, col_name, n, ex_others=ex_others)

        if sort_dim:
            plotdf = PyFix.sva(plotdf, col_name)
    
        else:
            plotdf = PyFix.svd(plotdf, 'count')

        types = plotdf[col_name].unique()
        cmap = plt.get_cmap('tab20b')
        colors = {type_: cmap(i / len(types)) for i, type_ in enumerate(types)}
        colors['All Other Values'] = '#ccc'

        figsize = 14 if plotdf.shape[0] > 10 else 10

        fig, ax = plt.subplots(figsize=(figsize, (plotdf.shape[0]/2)))

        bar_collections = {}
        for i, row in plotdf.iterrows():
            # if pd.isna(row[col_name]) or row[col_name] == '':
            #     row[col_name] = 'Unknown'

            label = row[col_name]
            if pd.isna(label) or label == '':
                label = 'Unknown'
            # Truncate label if necessary
            label = (label[:max_label_length] + '...') if len(label) > max_label_length else label

            color = colors.get(row[col_name], 'grey')  # Default to 'grey' if key doesn't exist
            bar = None
            
            if vertical:
                bar = ax.bar(row[col_name], row['count'], color=color)
            else:
                bar = ax.barh(row[col_name], row['count'], color=color)

            if row[col_name] not in bar_collections:
                bar_collections[row[col_name]] = bar

        ax.set_title(f'Frequency by {PyFix.unsnake(col_name)}')

        # Set a fixed size for the y-axis area to align labels
        ax.figure.subplots_adjust(left=0.2)  # Adjust this value to increase or decrease y-axis label space

        ax.legend(title=PyFix.unsnake(col_name), handles=list(bar_collections.values()), 
            labels=list(bar_collections.keys()), loc='lower right', bbox_to_anchor=(1, 0))


        if not vertical:
            ax.invert_yaxis()  # Largest bars on top

        if vertical:
            plt.xticks(rotation=90)       
        
        plt.show()

        if return_df:
            return plotdf

    @staticmethod
    def hist20(df, col_bar):
        return PyFix.hist(df, col_bar, 20)
        
    @staticmethod
    def hist10(df, col_bar):
        return PyFix.hist(df, col_bar, 10)

    @staticmethod
    def hist7(df, col_bar):
        return PyFix.hist(df, col_bar, 7)

    @staticmethod
    def remove_duplicates_keep_order(seq):
        seen = set()
        seen_add = seen.add
        return [x for x in seq if not (x in seen or seen_add(x))]

    @staticmethod
    def create_readable_bin_labels(bin_edges):
        """
        Generates readable labels for bins based on provided bin edges.
        
        Parameters:
            bin_edges (list): A list of numerical edges for bins.
        
        Returns:
            list: A list of strings representing bin labels, formatted as 'start-end', with the last bin labeled as 'Over X'.
        """
        labels = []
        for start, end in zip(bin_edges[:-1], bin_edges[1:]):
            if int(start) == int(end) - 1:
                labels.append(f"{int(start)}")
            else:
                labels.append(f"{int(start)}-{int(end)}")
        labels[-1] = f"Over {int(bin_edges[-2])}"
        return labels

    @staticmethod
    def bucket(df, col, bins=[0, 1, 2, 10, 100, 500, 1000, 10000, 50000, 100000 , 500000]):
        """
        Creates bins for a column in a DataFrame based on quantiles and provided bins,
        adjusting bins to include common values and account for data distribution.

        Parameters:
            df (DataFrame): The DataFrame containing the column to bin.
            col (str): The column name to bin.
            bins (list): Initial list of bin edges, which may be adjusted based on data.

        Modifies:
            df (DataFrame): Adds a new column named '{col}_bucket' with categorical bin labels.
        """
        p99 = int(df[col].quantile(.99))  # 99th percentile
        bins = [n for n in bins if n <= p99] + [p99 + 1]  # Adjust bins to 99th percentile

        # Adjust bin starting points based on the 5th percentile of positive values
        p05 = int(df.query(f'{col} > 0')[col].quantile(.05))
        if p05 > 100:
            bins = [n for n in bins if n != 2]

        # If negative values exist, ensure the minimum value is included in the bins
        min_value = df[col].min()
        if min_value < 0:
            bins.append(min_value)
            bins.sort()  # Sort the bins to maintain order after adding min_value

        labels = PyFix.create_readable_bin_labels(bins)
        # Adds on the new column with the buckets
        df[f'{col}_bucket'] = pd.cut(df[col], bins=bins, labels=labels, right=False)

    @staticmethod
    def breakdown(df, col_bar, col_color, n_bars=8, n_colors=15, ex_mode=0, vertical=1, perc=False, sort_mode=0):
        """
        Produces a horizontal or vertical bar chart with segments colored according to a secondary categorical variable.
        It displays the top 'n' categories in both the primary and secondary columns, with the ability
        to include an 'All Other Values' category for remaining data. Optionally, the bars can represent
        the percentage of each color category within each bar.
        
        Parameters:
        - df: DataFrame containing the data for plotting.
        - col_bar: The name of the primary categorical column to create bars for.
        - col_color: The name of the secondary categorical column to color segments within the bars.
        - n_bars: The number of top categories from col_bar to display.
        - n_colors: The number of top categories from col_color to display.
        - ex_mode: Mode of exclusions. 
            0: Display 'All Other Values' for both bar and color categories.
            1: Exclude 'All Other Values' for color categories only.
            2: Exclude 'All Other Values' for bar categories only.
            10: Exclude 'All Other Values' entirely.
        - percentage: If True, bars represent the percentage of each color category within each bar.
        - vertical: If 0 horizontal bars, if 1 vertical bars
        
        Returns:
        - None: The function directly plots the bar chart.
        
        Example usage:
        breakdown(rem, 'location_query', 'occupation', n_bars=7, ex_mode=10, percentage=True, vertical=0)
        This will plot the top 7 locations by occupation as a percentage distribution within each location.
        """
        # Get value counts and sort by col_bar
        plotdf = df.groupby([col_bar, col_color], observed=False).size().reset_index(name='count')

        if is_categorical_dtype(plotdf[col_bar]):
           plotdf[col_bar] = plotdf[col_bar].astype(str)

        if is_categorical_dtype(plotdf[col_color]):
           plotdf[col_color] = plotdf[col_color].astype(str)
        
        # Determine the top 'n' bars
        top_bars = plotdf.groupby(col_bar, observed=False)['count'].sum().nlargest(n_bars).index
        top_colors = plotdf.groupby(col_color, observed=False)['count'].sum().nlargest(n_colors).index
        
        if ex_mode == 1 or ex_mode == 10:
            plotdf = plotdf[plotdf[col_bar].isin(top_bars)]
            
        if ex_mode == 2 or ex_mode == 10:
            plotdf = plotdf[plotdf[col_color].isin(top_colors)]
        
        if ex_mode == 2 or ex_mode == 0:
            # Add all other values to dataframe for both bars and colors after N and n_col
            plotdf.loc[~plotdf[col_bar].isin(top_bars), col_bar] = 'All Other Values'
            top_bars = top_bars.to_list()
            top_bars.append('All Other Values')
            
        if ex_mode == 1 or ex_mode == 0:   
            plotdf.loc[~plotdf[col_color].isin(top_colors), col_color] = 'All Other Values'
            top_colors = top_colors.to_list()
            top_colors.append('All Other Values')

        # Make it smaller before processing
        plotdf = plotdf.groupby([col_bar, col_color], observed=False).sum().reset_index()

        # Now the df is filtered, we need to find the top bars and top colors to use for ordering.
        if sort_mode == 0:
            top_bars = plotdf.groupby(col_bar, observed=False)['count'].sum().nlargest(n_bars).index.to_list()

        if sort_mode == 10:
            top_colors = PyFix.sva(plotdf, col_color)[col_color].to_list()
            top_colors = PyFix.remove_duplicates_keep_order(top_colors)
            top_colors = [n for n in top_colors if n != 'All Other Values']
            top_colors.append('All Other Values')

            top_bars = PyFix.sva(plotdf, col_bar)[col_bar].to_list()
            top_bars = PyFix.remove_duplicates_keep_order(top_bars)
            top_bars = [n for n in top_bars if n != 'All Other Values']
            top_bars.append('All Other Values')

        # Sort by col_bar and then by the order you want for col_color
        color_order = top_colors
        
        plotdf[col_color] = pd.Categorical(plotdf[col_color], categories=color_order, ordered=True)
        plotdf = plotdf.sort_values(by=[col_bar, col_color])

        # Create the plot
        height = 5
        if not vertical:
            height = len(top_bars)/1.5
            
        fig, ax = plt.subplots(figsize=(9, height))

        # Initialize a dictionary to store the starting point of each bar segment
        y_starts = dict.fromkeys(top_bars, 0)

        # Create a color map for each unique value in the col_color column
        cmap = plt.get_cmap('tab20b')
        colors = {color: cmap(i / len(color_order)) for i, color in enumerate(color_order)}
        colors['All Other Values'] = '#ccc'
        
        # If percentage mode is on, normalize the counts within each bar to sum up to 1 (100%)
        if perc:
            # Normalize the counts within each group (col_bar)
            sum_counts = plotdf.groupby(col_bar, observed=False)['count'].transform('sum')
            plotdf['count'] = plotdf['count'] / sum_counts

        # Plot each segment of the bars
        for bar in top_bars:
            for i, row in plotdf[plotdf[col_bar] == bar].iterrows():
                color = colors.get(row[col_color], 'grey')  # Use 'grey' for undefined colors
                if vertical:
                    ax.bar(row[col_bar], row['count'], bottom=y_starts[row[col_bar]], color=color)
                else:
                    ax.barh(row[col_bar], row['count'], left=y_starts[row[col_bar]], color=color)
                
                y_starts[row[col_bar]] += row['count']  # Update the starting point for the next segment

        if not vertical:
            ax.invert_yaxis()  # Largest bars on top
            
        chart_type = 'Percentage' if perc else 'Count'
        ax.set_title(f'{chart_type} by {PyFix.unsnake(col_bar)} & {PyFix.unsnake(col_color)}')
        
        # Create custom legend
        handles = [plt.Rectangle((0, 0), 1, 1, color=colors[color]) for color in color_order]
        
        if vertical:
            plt.xticks(rotation=90)

        ax.legend(handles, color_order, title=PyFix.unsnake(col_color), loc='upper right', bbox_to_anchor=(2, 1))
        plt.show()
        
    @staticmethod
    def perc(df, extended=True):
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

        if extended:
            return pd.concat([x, perc_df], sort=False).round(0).apply(lambda x: x.astype(int) if not x.isnull().any() else x)
        else:
            return perc_df

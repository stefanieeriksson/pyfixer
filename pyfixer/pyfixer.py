#!/usr/bin/env python
# coding: utf-8

import re
from datetime import date

class PyFix:
    def __init__(self):
        pass

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
        """Returns a list of items in list_a that are not in exclude_these using set operations."""
        set_a = set(list_a)
        set_exclude = set(exclude_these)
        return list(set_a - set_exclude)

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

    @staticmethod
    def remove_duplicates_keep_order(seq):
        seen = set()
        seen_add = seen.add
        return [x for x in seq if not (x in seen or seen_add(x))]
         
    # General shorthand functions #
    @staticmethod
    def to_int(value, default=0):
        """try return int(value) except return default """
        try:
            if value:
                value = value.replace(',','').strip()
                if value.strip().lower().endswith('k'):
                    # Remove the 'k' and convert the remaining number to float, then multiply by 1000
                    number = float(value[:-1]) * 1000
                    return int(number)
                elif value.strip().lower().endswith('m'):
                    number = float(value[:-1]) * 1000000
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

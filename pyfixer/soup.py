#!/usr/bin/env python
# coding: utf-8

from bs4 import BeautifulSoup as bsoup
import re

class SoupHelper:
    def __init__(self):
        pass

    def get_soup(soup, elem):
        return soup.find_all(elem)

    def show_soup(soup, elem):
        for a in soup.find_all(elem):
            print(a)
            print('---------------------------')
            print()

    def class_soup(soup, elem, classname):
        return soup.find_all(elem, attrs={'class': classname})

    def get_a(soup):
        return soup.find_all('a')

    def class_a(soup, classname):
        return soup.find_all('a', attrs={'class': classname})

    def show_a(soup):
        for a in soup.find_all('a'):
            print(a.text, a['href'])

    def get_div(soup):
        return soup.find_all('div')

    def div_class(soup, classname):
        return soup.find_all('div', attrs={'class': classname})

    def show_div_class(soup, classname):
        i = 0
        for a in soup.find_all('div', attrs={'class': classname}):
            print(a)
            print('---------------------------' , i)
            print()
            i = i + 1

    def show_div_class_text(soup, classname):
        i = 0
        for a in soup.find_all('div', attrs={'class': classname}):
            if a.text:
                print(re.sub(r'\s+', ' ', a.text.strip()))
                print('---------------------------' , i)
                print()
            i = i + 1

    def show_div(soup):
        for a in soup.find_all('div'):
            print(a)
            print('---------------------------')
            print()

    def show_div_text(soup):
        i = 0
        for a in soup.find_all('div'):
            if a.text:
                print(re.sub(r'\s+', ' ', a.text.strip()))
                print('---------------------------' , i)
                print()
                i = i + 1

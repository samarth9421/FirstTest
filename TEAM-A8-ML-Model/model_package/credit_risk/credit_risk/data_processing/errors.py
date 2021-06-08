# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 05:28:53 2021

@author: Team A8
"""

class BaseError(Exception):
    """Base package error."""


class InvalidInputError(BaseError):
    """input contains an error."""

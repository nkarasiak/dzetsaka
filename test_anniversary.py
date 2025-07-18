#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for anniversary popup functionality (without GUI dependencies)
"""

import datetime


def test_anniversary_logic():
    """Test the anniversary logic without GUI dependencies"""
    print("Testing Anniversary Logic...")
    
    # Test date logic
    ANNIVERSARY_DATE = datetime.date(2026, 5, 17)  # May 17, 2026
    FEATURE_COLLECTION_START = datetime.date(2025, 7, 17)  # Start collecting features from now
    today = datetime.date.today()
    
    print(f"Today's date: {today}")
    print(f"Feature collection start: {FEATURE_COLLECTION_START}")
    print(f"Anniversary date: {ANNIVERSARY_DATE}")
    
    # Test if we're in the feature collection period
    in_collection_period = FEATURE_COLLECTION_START <= today <= ANNIVERSARY_DATE
    print(f"In feature collection period: {in_collection_period}")
    
    if in_collection_period:
        # Test date calculation
        days_until_anniversary = (ANNIVERSARY_DATE - today).days
        print(f"Days until anniversary: {days_until_anniversary}")
        
        if days_until_anniversary > 0:
            print(f"[FEATURE COLLECTION] Anniversary is in {days_until_anniversary} days")
            print("[SUCCESS] POPUP SHOULD BE SHOWN")
        elif days_until_anniversary == 0:
            print("[ANNIVERSARY] Today is the anniversary!")
            print("[SUCCESS] POPUP SHOULD BE SHOWN")
        else:
            print(f"Anniversary was {abs(days_until_anniversary)} days ago")
    else:
        if today < FEATURE_COLLECTION_START:
            days_until_start = (FEATURE_COLLECTION_START - today).days
            print(f"Feature collection starts in {days_until_start} days")
        else:
            days_since_anniversary = (today - ANNIVERSARY_DATE).days
            print(f"Anniversary ended {days_since_anniversary} days ago")
        print("[FAIL] POPUP SHOULD NOT BE SHOWN")
    
    print("Anniversary logic test completed!")


if __name__ == "__main__":
    test_anniversary_logic()
    print("\nTest completed successfully!")
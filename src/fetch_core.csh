#!/bin/tcsh -f

# Ensure the TARGET_DATE is at least 7 days in the past
set TARGET_DATE = "20260920"

foreach CYCLE (00 06 12 18)
    set DATE_STAMP = "${TARGET_DATE}${CYCLE}"
    echo "--- Pulling 37 Levels for Cycle: $DATE_STAMP ---"

    # Step 1: Pressure Levels
    # Fix: Added '.' to the regex to capture decimal levels like 0.4 mb
    python3 get_core.py pgb $DATE_STAMP 1 1 ':(TMP|UGRD|VGRD):' . 
    # Step 2: Surface Fields
    #echo "Step 2: Fetching Surface Fields..."
    python3 get_core.py flx $DATE_STAMP 1 1 ':(PRES|TMP|UGRD|VGRD):((surface)|(10 m above ground))' .
    
    echo "--- Done with $DATE_STAMP ---"
end



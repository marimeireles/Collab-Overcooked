#!/bin/bash

# Script to partition runs_lvl2_symmetric.csv into smaller chunks
# Each chunk will have 3 lines of content plus header

LINES_PER_CHUNK=3
GLOBAL_COUNTER=1
INPUT_CSV="final_for_real.csv"

# Check if input file exists
if [ ! -f "$INPUT_CSV" ]; then
    echo "Error: File '$INPUT_CSV' not found!"
    exit 1
fi

echo "Processing $INPUT_CSV..."

# Get the header (first line)
HEADER=$(head -n 1 "$INPUT_CSV")

# Count total data lines (excluding header and empty lines)
TOTAL_DATA_LINES=$(tail -n +2 "$INPUT_CSV" | grep -v '^[[:space:]]*$' | wc -l)

if [ $TOTAL_DATA_LINES -eq 0 ]; then
    echo "Error: No data lines found in '$INPUT_CSV'!"
    exit 1
fi

echo "  - Data lines: $TOTAL_DATA_LINES"

# Calculate number of chunks needed for this file
CHUNKS_NEEDED=$(( (TOTAL_DATA_LINES + LINES_PER_CHUNK - 1) / LINES_PER_CHUNK ))
echo "  - Chunks to create: $CHUNKS_NEEDED"

# Create temporary file with only data lines (no header, no empty lines)
TEMP_DATA_FILE=$(mktemp)
tail -n +2 "$INPUT_CSV" | grep -v '^[[:space:]]*$' > "$TEMP_DATA_FILE"

# Split the data into chunks
for (( i=1; i<=CHUNKS_NEEDED; i++ )); do
    OUTPUT_FILE="final_for_real_${GLOBAL_COUNTER}.csv"
    
    # Calculate start and end lines for this chunk
    START_LINE=$(( (i-1) * LINES_PER_CHUNK + 1 ))
    END_LINE=$(( i * LINES_PER_CHUNK ))
    
    # Create the output file with header
    echo "$HEADER" > "$OUTPUT_FILE"
    
    # Add the data lines for this chunk
    sed -n "${START_LINE},${END_LINE}p" "$TEMP_DATA_FILE" >> "$OUTPUT_FILE"
    
    # Count actual lines added
    ACTUAL_LINES=$(tail -n +2 "$OUTPUT_FILE" | wc -l)
    
    echo "  - Created $OUTPUT_FILE with $ACTUAL_LINES data lines"
    
    GLOBAL_COUNTER=$((GLOBAL_COUNTER + 1))
done

# Clean up temporary file
rm "$TEMP_DATA_FILE"

echo "  - Finished processing $INPUT_CSV"
echo
echo "Partitioning complete!"
echo "Created $(($GLOBAL_COUNTER - 1)) files: final_for_real_1.csv to final_for_real_$(($GLOBAL_COUNTER - 1)).csv"


#!/bin/bash

# Check if failed chunk files exist

ls -la data/videos/FAILED_CHUNK_*.mp4

# Create a file list for ffmpeg concat                                                                                                                                   
echo "Creating file list..."                                                                                                                                             
for f in data/videos/FAILED_CHUNK_*.mp4; do                                                                                                                              
    if [ -f "$f" ]; then                                                                                                                                                 
        echo "file '$PWD/$f'" >> /tmp/filelist.txt                                                                                                                       
    fi                                                                                                                                                                   
done  
echo "File list contents:"                                                                                                                                                 
cat /tmp/filelist.txt                                                                                                                                                      

# Concatenate the files                                                                                                                                                    
echo "Concatenating videos..."                                                                                                                                             
ffmpeg -f concat -safe 0 -i /tmp/filelist.txt -c copy combined_failed_chunks.mp4                                                                                           

# Clean up                                                                                                                                                                 
rm /tmp/filelist.txt  
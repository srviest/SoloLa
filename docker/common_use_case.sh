# env: windows git bash
# execution location: solola/
# input audio location: locate at same directory with beat_tracking.py, e.g. test.mp3 or input/test.mp3
# output: output/[filename]/FinalNotes.txt e.g. 　output/test/FinalNotes.txt 
docker run -ti --rm -v E:\workplace\projects\solola\solola:/solola solola-py35:prod2 python3 main.py test.mp3

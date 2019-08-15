# env: windows cmd.exe
# execution location: solola/
# input audio location: locate at same directory with main.py, e.g. test.mp3 or input/test.mp3
# output: output/[filename]/FinalNotes.txt e.g. output/test/FinalNotes.txt 

# image: solola-py35:prod
docker run -ti --rm -v E:\workplace\projects\solola\solola\inputs:/solola/inputs -v E:\workplace\projects\solola\solola\outputs:/solola/outputs solola-py35:prod2 python3 main.py inputs/test.mp3

# image: solola-py35:dev
docker run -ti --rm -v E:\workplace\projects\solola\solola\inputs:/solola/inputs -v E:\workplace\projects\solola\solola\outputs:/solola/outputs solola-py35:dev python3 main.py inputs/test.mp3
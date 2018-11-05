# run a simple test case to valildate SoloLa-py3 working well
project_root=$(pwd)
test_data="$project_root/test_input/Bohemian_Rhapsody_SOLO_Guitarraviva_normal_speed.mp3"
echo $test_data
python3 main.py $test_data 

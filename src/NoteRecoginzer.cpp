/*
NoteRecognizer

YUAN-PING CHEN 
2015/07/14
Usage: 
    $ ./NoteRecognizer Melody.txt Note_output_path.txt Note_in_frame_output_path.txt

Input:  1. Frame-wise pitch contour in Hertz 
        2. Path of estimated note-wise sequence
        3. Path of estimated frame-wise note sequence

Output: Estimated note-wise sequence
        Estimated frame-wise note sequence
*/

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include "NoteRecognizer.h"
using std::vector;
using std::pair;
using std::cout;
using std::endl;
using std::ofstream;
using std::stringstream;
using std::fstream;
using std::string;
using std::ios;

int main(int argc, char** argv){
    // ARGUMENT EXAMINAITON
    if (argc<2){
        cout << "You need to supply one argument to this program.";
        return -1;
    }
    

    // LOAD PITCH
    std::fstream fs;
    fs.open(argv[1],std::ios::in);
    std::vector<std::vector<float> > v;
    std::string strFloat;
    float fNum;
    int counter = 0;
    while (getline(fs, strFloat)){
        std::stringstream linestream(strFloat);
        
        v.push_back(std::vector<float>());
        
        while (linestream >> fNum)
        {
            v[counter].push_back(fNum);
        }
        ++counter;
    }

    //
    std::vector<float> mpOut;
    for (int i = 0; i < v.size(); ++i){
        mpOut.push_back(v[i][0]);
    }
  

    // MONO-NOTE STUFF
    //    std::cerr << "Mono Note Stuff" << std::endl;
    MonoNote mn;
    std::vector<std::vector<std::pair<double, double> > > smoothedPitch;
    for (size_t iFrame = 0; iFrame < mpOut.size(); ++iFrame) {
        std::vector<std::pair<double, double> > temp;
        if (mpOut[iFrame] > 0)
        {
            double tempPitch = 12 * std::log(mpOut[iFrame]/440)/std::log(2.) + 69; //hertz to MIDI number 
            temp.push_back(std::pair<double,double>(tempPitch, .9));
        }
        smoothedPitch.push_back(temp);
    }

    // std::cout << smoothedPitch. << std::endl;
    // vector<MonoNote::FrameOutput> mnOut = mn.process(m_pitchProb);
    vector<MonoNote::FrameOutput> mnOut = mn.process(smoothedPitch); // 
    
    // ---------------------------------------------be to turn on---------------------------------------------
    float values = 0;
    float timestamp = 0;
    float duration = 0;
    int onsetFrame = 0;
    bool isVoiced = 0;
    bool oldIsVoiced = 0;
    size_t nFrame = smoothedPitch.size();
    std::vector<float> Note;
    std::vector<vector<float> > Notes;
    float minNoteFrames = (m_inputSampleRate*m_pruneThresh) / m_stepSize;
    // std::cout << "nFrame is" << nFrame << "\n" << std::endl;
    std::vector<float> notePitchTrack; // collects pitches for one note at a time
    for (size_t iFrame = 0; iFrame < nFrame; ++iFrame)
    // for (size_t iFrame = 0; iFrame < 100; ++iFrame)
    {
        //CHECK IF THE FRAME IS VOICED
        isVoiced = mnOut[iFrame].noteState < 3
                   && smoothedPitch[iFrame].size() > 0;
                   // && (iFrame >= nFrame-2);
                       // || ((m_level[iFrame]/m_level[iFrame+2]) > m_onsetSensitivity));  //m_level: average amplitude (root mean suqare) of a frame
        // std::cout << "iFrame is " << iFrame << std::endl;
        // std::cout << "isVoiced is " << isVoiced << std::endl;
        // std::cout << "smoothedPitch[iFrame].size() is " << smoothedPitch[iFrame].size() << std::endl;
        // std::cout << "notePitchTrack.size() is " << notePitchTrack.size() << std::endl;
        if (isVoiced && iFrame != nFrame-1) //The frame is voiced and not the end of the track
        {
            if (oldIsVoiced == 0) // beginning of a note
            {
                onsetFrame = iFrame;
                // std::cout << "onsetFrame is " << onsetFrame << std::endl;
                // std::cout << "m_stepSize is " << m_stepSize << std::endl;
                // std::cout << "m_inputSampleRate is " << m_inputSampleRate << std::endl;
                // std::cout << "m_stepSize/m_inputSampleRate is " << m_stepSize/m_inputSampleRate << std::endl;
            }
            float pitch = smoothedPitch[iFrame][0].first; // pitch of iFrame
            notePitchTrack.push_back(pitch); // add to the note's pitch track
        } 
        else 
        { // not currently voiced
            if (oldIsVoiced == 1) // end of note
            {
                // std::cerr << notePitchTrack.size() << " " << minNoteFrames << std::endl;
                if (notePitchTrack.size() >= minNoteFrames)
                {
                    std::sort(notePitchTrack.begin(), notePitchTrack.end());
                    float medianPitch = notePitchTrack[notePitchTrack.size()/2];
                    medianPitch = round(medianPitch);
                    // float medianFreq = std::pow(2,(medianPitch - 69) / 12) * 440;
                    Note.clear();
                    Note.push_back(medianPitch);
                    Note.push_back(onsetFrame*m_stepSize/m_inputSampleRate);
                    Note.push_back(float((iFrame * m_stepSize / m_inputSampleRate) - (onsetFrame * m_stepSize / m_inputSampleRate)));
                    Notes.push_back(Note);
                    std::cout <<  Note[0] << " " << Note[1] << " " << Note[2] <<std::endl;
                }
                notePitchTrack.clear();
            }
        }
        oldIsVoiced = isVoiced;
        // std::cout << "\n" << std::endl;
    }
    // ---------------------------------------------be to turn on---------------------------------------------
    
    // std::cout << argv[1] << " has "<< Notes.size() << " notes." << std::endl;
    
    // //stdout data 
    // for (int i=0; i < v.size(); ++i){
    //     std::cout << v[i][0] << " " << v[i][1] << " " << v[i][2] << std::endl;
    // }
    if (argc>=3){
        //SAVE NOTES LEVEL AS TEXT FILE
        string filename2 = argv[2];
        std::ofstream of2(filename2);
        for (int iNote=0; iNote < Notes.size(); iNote++){
            of2 << Notes[iNote][0] << " ";
            of2 << Notes[iNote][1] << " ";
            of2 << Notes[iNote][2] << " ";
            of2 << "\n"; 
        }

        if (argc>3){
            // //SAVE NOTES IN FRAME LEVEL AS TEXT FILE
            string filename3 = argv[3];
            std::ofstream of3(filename3);
            for (int iFrame=0; iFrame<v.size(); iFrame++){
                of3 << mnOut[iFrame].frameNumber << " ";
                of3 << mnOut[iFrame].pitch << " ";
                of3 << mnOut[iFrame].noteState << " ";
                of3 << "\n";
            }
        }
    } 
    // //SAVE NOTES LEVEL AS TEXT FILE
    // string filename2 = argv[2];
    // std::ofstream of2(filename2);
    // for (int iNote=0; iNote < Notes.size(); iNote++){
    //     of2 << Notes[iNote][0] << " ";
    //     of2 << Notes[iNote][1] << " ";
    //     of2 << Notes[iNote][2] << " ";
    //     of2 << "\n"; 
    // }
    // // //SAVE NOTES IN FRAME LEVEL AS TEXT FILE
    // string filename3 = argv[3];
    // std::ofstream of3(filename3);
    // for (int iFrame=0; iFrame<v.size(); iFrame++){
    //     of3 << mnOut[iFrame].frameNumber << " ";
    //     of3 << mnOut[iFrame].pitch << " ";
    //     of3 << mnOut[iFrame].noteState << " ";
    //     of3 << "\n";
    // }

    return 0;
}
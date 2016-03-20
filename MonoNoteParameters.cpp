/* -*- c-basic-offset: 4 indent-tabs-mode: nil -*-  vi:set ts=8 sts=4 sw=4: */

/*
    pYIN - A fundamental frequency estimator for monophonic audio
    Centre for Digital Music, Queen Mary, University of London.
    
    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU General Public License as
    published by the Free Software Foundation; either version 2 of the
    License, or (at your option) any later version.  See the file
    COPYING included with this distribution for more information.
*/

#include "MonoNoteParameters.h"

MonoNoteParameters::MonoNoteParameters() :
    // initializer list
    minPitch(38),            // D2 = 38
    nPPS(1),                 // number of pitches per semitone
    nS(52),                  // 38+52=90=F6#
    nSPP(3),                 // states per pitch
    n(0),                    // number of states (will be calcualted from other parameters)
    initPi(0),               // initial state probabilities

    pAttackSelftrans(0.4),   // transition probibility of attack to attack
    pStableSelftrans(0.4),  // transition probibility of stable to stable
    pSilentSelftrans(0.4),// transition probibility of silent to silent

    sigma2Note(5),           // sigma of next note Gaussian distribution
    maxJump(13),             // maximal distance for a transition in semitone
    minSemitoneDistance(0.5), // minimal distance for a transition in semitone

    priorPitchedProb(.7),    // parameters for voiced/unvoiced 
    priorWeight(0.7),        // voicing probability

    sigmaYinPitchAttack(1.1),// 3.2 sigma of observation probability distribution
    sigmaYinPitchStable(0.8),// 1.2 sigma of observation probability distribution
    sigmaYinPitchSilent(1.1),// 2.2 sigma of observation probability distribution
    yinTrust(0.1)            // tau: how much the pitch estimate is trusted
{
    // just in case someone put in a silly value for pRelease2Unvoiced
    n = nPPS * nS * nSPP;
}

MonoNoteParameters::~MonoNoteParameters()
{
    // // // Origin initializer list
    // minPitch(38),            // D2 = 38
    // nPPS(1),                 // number of pitches per semitone
    // nS(52),                  // 38+52=90=F6#
    // nSPP(3),                 // states per pitch
    // n(0),                    // number of states (will be calcualted from other parameters)
    // initPi(0),               // initial state probabilities

    // pAttackSelftrans(0.9),   // transition probibility of attack to attack
    // pStableSelftrans(0.99),  // transition probibility of stable to stable
    // pSilentSelftrans(0.9999),// transition probibility of silent to silent

    // sigma2Note(0.7),         // sigma of next note Gaussian distribution
    // maxJump(13),             // maximal distance for a transition in semitone
    // minSemitoneDistance(.5), // minimal distance for a transition in semitone

    // priorPitchedProb(.7),    // parameters for voiced/unvoiced 
    // priorWeight(0.7),        // voicing probability

    // sigmaYinPitchAttack(5),  // sigma of observation probability distribution
    // sigmaYinPitchStable(0.8),// sigma of observation probability distribution
    // sigmaYinPitchSilent(1.0),// sigma of observation probability distribution
    // yinTrust(0.1)            // tau: how much the pitch estimate is trusted

}
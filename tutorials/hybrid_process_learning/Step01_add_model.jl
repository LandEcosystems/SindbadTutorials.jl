# this script needs to be run in the terminal, not in the REPL
using Revise
using SindbadTutorials

generateSindbadApproach(:gppAirT, 
                        "Effect of temperature on GPP: 1 indicates no temperature stress, 0 indicates complete stress.", 
                        :externalNN, 
                        "Use external ML model", 
                        1)

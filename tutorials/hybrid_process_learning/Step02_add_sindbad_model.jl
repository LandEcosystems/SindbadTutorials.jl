using Revise
using SindbadTutorials

# the calls to the function need to be copy pasted into the REPL and not using Shift+Enter to run
# Note that force_over_write=:approach is used to overwrite the approach if it already exists. So, use with caution.

generateSindbadApproach(:gppAirT, 
                               "Effect of temperature on GPP: 1 indicates no temperature stress, 0 indicates complete stress.", 
                               :externalNN, 
                               "Use continuous external ML model", 
                               1,
                               methods=(),
                               force_over_write=:approach)


generateSindbadApproach(:gppAirT, 
                               "Effect of temperature on GPP: 1 indicates no temperature stress, 0 indicates complete stress.", 
                               :externalNN_LUT, 
                               "Use lookup table from external ML model", 
                               1,
                               methods=(),
                               force_over_write=:approach)
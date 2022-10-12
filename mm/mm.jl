# The naming of parameters follows Goutsias (2005)
rs = @reaction_network begin
    c2 * (s - P - (e - E)), E --> 0
    c3 * (e - E), 0 --> E
    c1 * (e - E), 0 --> E + P
end c1 c2 c3 e s

rs_qea = @reaction_network begin
    c1 * (min(s, e + P) - P), 0 --> P
end c1 c2 c3 e s

rs_qssa = @reaction_network begin
    c1 * e * (s - P) / ((c3 + c1) / c2 + s - P), 0 --> P
end c1 c2 c3 e s

sys = HiddenFSPSystem(rs, DefaultIndexHandler(), [ 0 1 ], [ 1 0 ]);

E0 = 10
S0 = 100

gt_params = [ 0.1, 1, 1, E0, S0 ]
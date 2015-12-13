local _ = require 'moses'
local gnuplot = require 'gnuplot'
local environ = require 'environ'

local nEpisodes = 100000
-- Number of discrete actions
local m = #environ.A

-- No discounting
local gamma = 1
-- Action-value function
local Q = torch.zeros(10, 21, m)
-- Number of times a state is visited per action
local N = torch.zeros(10, 21, m)
local NZero = 100

-- Sample
for i = 1, nEpisodes do
  -- Experience tuples (s, a, r)
  local E = {}
  -- Pick random starting state
  local s = {torch.random(1, 10), torch.random(1, 21)}
  
  -- Run till termination
  repeat
    -- Calculate (time-varying) epsilon dependent on state visits
    local epsilon = NZero/(NZero + torch.sum(N[s[1]][s[2]]))

    -- Choose action by epsilon-greedy exploration
    local aIndex
    if torch.uniform() < (1 - epsilon) then
      -- Pick argmax action with probability 1 - epsilon
      __, aIndex = torch.max(N[s[1]][s[2]], 1)
      aIndex = aIndex[1]
    else
      -- Otherwise pick any action with probability 1/m
      aIndex = torch.random(1, m)
    end
    local a = environ.A[aIndex]
    
    -- Increment state counter (every-visit Monte-Carlo policy evaluation)
    N[s[1]][s[2]][aIndex] = N[s[1]][s[2]][aIndex] + 1

    -- Perform a step
    local sPrime, r = environ.step(s, a)

    -- Store experience tuple
    table.insert(E, {s, a, r})

    -- Set next state as current state
    s = sPrime
  until environ.isTerminal(a, r)

  -- Learn from experience of one complete episode (hence only works in episodic problems)
  for j = 1, #E do
    -- Calculate time-dependent return
    local G = 0
    local Gt = 0
    for t = j, #E do
      G = G + math.pow(gamma, Gt)*E[t][3]
      Gt = Gt + 1
    end

    -- Extract experience
    local s = E[j][1]
    local a = E[j][2]
    -- Get action index
    local aIndex = _.find(environ.A, a)

    -- Calculate (time-varying) step size...
    local alpha = 1/N[s[1]][s[2]][aIndex]

    -- Estimate value by mean return
    Q[s[1]][s[2]][aIndex] = Q[s[1]][s[2]][aIndex] + alpha*(G - Q[s[1]][s[2]][aIndex])
  end
end

-- Extract V as argmax Q
local V = torch.max(Q, 3):squeeze()
-- Plot V
gnuplot.pngfigure('V.png')
gnuplot.splot(V)
gnuplot.title('V*')
gnuplot.ylabel('Player sum')
gnuplot.xlabel('Dealer showing')
gnuplot.plotflush()

-- Save Q
torch.save('Q.t7', Q)

local gnuplot = require 'gnuplot'
local step = require 'step'

-- Calculates if the current state is terminal given previous action and reward
local isTerminal = function(a, r)
  if a == 'hit' and r ~= -1 then
    return false
  else
    return true
  end
end

local nSamples = 100000
-- Discrete (indexed) actions
local A = {'hit', 'stick'}
-- Number of discrete actions
local m = #A
-- Action-value as a function of dealer's first card and player's sum
local Q = torch.zeros(10, 21, m)
-- Number of times a state is visited per action
local N = torch.zeros(10, 21, m)
local NZero = 100
-- Total return
local S = torch.zeros(10, 21, m)

local s, a, r, sPrime, epsilon

-- Sample
for i = 1, nSamples do
  -- Pick random starting state
  s = {torch.random(1, 10), torch.random(1, 21)}
  repeat
    -- Calculate (time-varying) epsilon dependent on state visits
    epsilon = NZero/(NZero + torch.sum(N[s[1]][s[2]]))
    -- Choose action from epsilon-greedy exploration
    local aIndex
    if torch.uniform() < (1 - epsilon) then
      -- Pick argmax action with probability 1 - epsilon
      __, aIndex = torch.max(N[s[1]][s[2]], 1)
      aIndex = aIndex[1]
    else
      -- Otherwise pick any action with probability 1/m
      aIndex = torch.random(1, m)
    end
    a = A[aIndex]
    -- Perform a step
    sPrime, r = step(s, a)

    -- Increment state counter (every-visit Monte-Carlo policy evaluation)
    N[s[1]][s[2]][aIndex] = N[s[1]][s[2]][aIndex] + 1
    -- Increment total return
    S[s[1]][s[2]][aIndex] = S[s[1]][s[2]][aIndex] + r
    -- Estimate value by mean return
    Q[s[1]][s[2]][aIndex] = S[s[1]][s[2]][aIndex] / N[s[1]][s[2]][aIndex]

    -- Set next state as current state
    s = sPrime
  until isTerminal(a, r)
end

-- Extract V as argmax Q
local V = torch.max(Q, 3):squeeze()
-- Create plot
gnuplot.pngfigure('V.png')
gnuplot.splot(V)
gnuplot.title('V*')
gnuplot.ylabel('Player sum')
gnuplot.xlabel('Dealer showing')
gnuplot.plotflush()

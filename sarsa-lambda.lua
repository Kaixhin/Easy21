local gnuplot = require 'gnuplot'
local environ = require 'environ'

-- Load Q* from MC control
local QStar = torch.load('Q.t7')

local nEpisodes = 1000
-- Number of discrete actions
local m = #environ.A

-- No discounting
local gamma = 1
-- Action-value function
local Q = torch.zeros(10, 21, m)
-- Number of times a state is visited per action
local N = torch.zeros(10, 21, m)
local NZero = 100

-- Sarsa(lambda) errors versus Q*
local lambdaErrors = {}

-- Compare different values of lambda
for lambda = 0, 1, 0.1 do
  -- Learning curve
  local epLambdaErrors = {}

  -- Sample
  for i = 1, nEpisodes do
    -- Eligibility traces for backward view Sarsa(lambda)
    local El = torch.zeros(10, 21, m)
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
      
      -- Increment state and eligibility counters
      N[s[1]][s[2]][aIndex] = N[s[1]][s[2]][aIndex] + 1
      El[s[1]][s[2]][aIndex] = gamma*lambda*El[s[1]][s[2]][aIndex] + 1

      -- Perform a step
      local sPrime, r = environ.step(s, a)

      local aPrimeIndex, delta
      if sPrime[2] >= 1 and sPrime[2] <= 21 then
        -- Choose action greedily for sPrime
        __, aPrimeIndex = torch.max(N[sPrime[1]][sPrime[2]], 1)
        aPrimeIndex = aPrimeIndex[1]

        -- Calculate TD-error
        delta = r + gamma*Q[sPrime[1]][sPrime[2]][aPrimeIndex] - Q[s[1]][s[2]][aIndex]
      else
        -- In terminal states, Q(s', a') = 0
        delta = r - Q[s[1]][s[2]][aIndex]
      end

      -- Calculate (time-varying) step size
      local alpha = 1/N[s[1]][s[2]][aIndex]
      -- Update Q
      Q[s[1]][s[2]][aIndex] = Q[s[1]][s[2]][aIndex] + alpha*delta*El[s[1]][s[2]][aIndex]

      -- Set next state as current state
      s = sPrime
    until environ.isTerminal(a, r)

    -- Keep learning curve for lambda = 0, 1 (doubles cannot be compared)
    if tostring(lambda) == '0' or tostring(lambda) == '1' then
      table.insert(epLambdaErrors, torch.sum(torch.pow((Q - QStar), 2)))
    end
  end

  -- Calculate error versus Q*
  local lambdaError = torch.sum(torch.pow((Q - QStar), 2))
  table.insert(lambdaErrors, lambdaError)

  -- Plot learning curve for lambda = 0, 1 (doubles cannot be compared)
  if tostring(lambda) == '0' or tostring(lambda) == '1' then
    gnuplot.pngfigure('SarsaLambda' .. lambda .. 'Learning.png')
    gnuplot.plot('Sq. Error', torch.linspace(1, 1000, 1000), torch.Tensor(epLambdaErrors), '-')
    gnuplot.title('Sarsa(' .. lambda .. ') learning curve')
    gnuplot.ylabel('Squared error versus Q*')
    gnuplot.xlabel('Episode #')
    gnuplot.plotflush()
  end
end

-- Plot Sarsa(lambda) errors
gnuplot.pngfigure('SarsaLambda.png')
gnuplot.plot('Sq. Error', torch.linspace(0, 1, 11), torch.Tensor(lambdaErrors))
gnuplot.title('Sarsa(lambda) errors')
gnuplot.ylabel('Squared error versus Q*')
gnuplot.xlabel('Lambda')
gnuplot.plotflush()

local gnuplot = require 'gnuplot'
local environ = require 'environ'

-- Load Q* from MC control
local QStar = torch.load('Q.t7')
-- NB: All values of λ diverge as episodes increase; λ = 1 diverges massively
local nEpisodes = 250
-- Number of discrete actions
local m = #environ.A

-- Translates state to coarse-coded feature vector (not incl. action)
local sToFeat = function(s)
  local ind = torch.zeros(3, 6):byte()

  -- Dealer indices
  local dInds = {}
  if s[1] >= 1 and s[1] <= 4 then
    table.insert(dInds, 1)
  end
  if s[1] >= 4 and s[1] <= 7 then
    table.insert(dInds, 2)
  end
  if s[1] >= 7 and s[1] <= 10 then
    table.insert(dInds, 3)
  end

  -- Player indices
  local pInds = {}
  if s[2] >= 1 and s[2] <= 6 then
    table.insert(pInds, 1)
  end
  if s[2] >= 4 and s[2] <= 9 then
    table.insert(pInds, 2)
  end
  if s[2] >= 7 and s[2] <= 12 then
    table.insert(pInds, 3)
  end
  if s[2] >= 10 and s[2] <= 15 then
    table.insert(pInds, 4)
  end
  if s[2] >= 13 and s[2] <= 18 then
    table.insert(pInds, 5)
  end
  if s[2] >= 16 and s[2] <= 21 then
    table.insert(pInds, 6)
  end

  -- Create index mask
  for d = 1, #dInds do
    for p = 1, #pInds do
      ind[dInds[d]][pInds[p]] = 1
    end
  end
  return ind
end

-- Translates state to an action-independent feature vector (or mask)
local sToMask = function(s)
  local feat = torch.zeros(3, 6, m):byte()
  local ind = sToFeat(s)
  for a = 1, m do
    feat[{{}, {}, {a}}] = ind
  end
  return feat
end

-- Translates state and action index to feature vector
local saToFeat = function(s, aIndex)
  local feat = torch.zeros(3, 6, m):byte()
  feat[{{}, {}, {aIndex}}] = sToFeat(s)
  return feat
end

-- Approximates Q
local calcQ = function(params)
  local Q = torch.Tensor(10, 21, m)

  local feat
  -- Loop over all action-state values
  for s1 = 1, 10 do
    for s2 = 1, 21 do
      for aInd = 1, m do
        feat = saToFeat({s1, s2}, aInd)
        Q[s1][s2][aInd] = torch.dot(feat:view(-1):double(), params)
      end
    end
  end

  return Q
end

-- No discounting
local gamma = 1
-- Parameters θ
local theta = torch.Tensor(3*6*m):normal(0, 0.01)
-- Number of times a state is visited per action
local N = torch.zeros(3, 6, m)
local NZero = 100

-- Constant exploration ɛ
local epsilon = 0.05
-- Constant step-size ɑ
local alpha = 0.01

-- λ errors versus Q*
local lambdaErrors = {}

-- Compare different values of λ
for lambda = 0, 1, 0.1 do
  -- Learning curve
  local epLambdaErrors = {}

  -- Sample
  for i = 1, nEpisodes do
    -- Eligibility traces for backward view Sarsa(λ)
    local El = torch.zeros(3, 6, m)
    -- Pick random starting state
    local s = {torch.random(1, 10), torch.random(1, 21)}

    -- Run till termination
    repeat
      -- Create (state-only) mask
      local mask = sToMask(s)
      -- Choose action by ɛ-greedy exploration
      local aIndex
      if torch.uniform() < (1 - epsilon) then
        -- Unroll approximated states to choose between the actions
        local states = torch.mean(N[mask]:view(m, -1), 2) -- TODO: Is taking the mean here appropriate?
        -- Pick argmax action with probability 1 - ɛ
        __, aIndex = torch.max(states, 1)
        aIndex = aIndex[1][1]
      else
        -- Otherwise pick any action with probability 1/m
        aIndex = torch.random(1, m)
      end
      local a = environ.A[aIndex]
      
      -- Calculate full feature
      local phi = saToFeat(s, aIndex)

      -- Perform a step
      local sPrime, r = environ.step(s, a)

      local aPrimeIndex, delta
      -- Create (state-only) mask for transition
      local maskPrime = sToMask(sPrime)
      if sPrime[2] >= 1 and sPrime[2] <= 21 then
        -- Unroll approximated states to choose between the actions
        local statesPrime = torch.mean(N[maskPrime]:view(m, -1), 2) -- TODO: Is taking the mean here appropriate?
        -- Choose action greedily for s'
        __, aPrimeIndex = torch.max(statesPrime, 1)
        aPrimeIndex = aPrimeIndex[1][1]
        -- Calculate full feature for transition
        local phiPrime = saToFeat(sPrime, aPrimeIndex)

        -- Calculate TD-error
        delta = r + gamma*torch.dot(phiPrime:view(-1):double(), theta) - torch.dot(phi:view(-1):double(), theta)
      else
        -- In terminal states, Q(s', a') = 0
        delta = r - torch.dot(phi:view(-1):double(), theta)
      end

      -- Increment state and eligibility counters
      N[phi] = N[phi] + 1
      -- NB: Updating with ɣ and λ here is inconsistent with Sarsa but consistent with pseudocode
      El:mul(gamma):mul(lambda):add(phi:double())

      -- Step-size x prediction error x eligbility trace x feature value (gradient of linear function)
      local dw = torch.cmul(phi:view(-1):double(), torch.mul(El, alpha*delta))
      -- Gradient descent
      theta:csub(dw)

      -- Set next state as current state
      s = sPrime
    until environ.isTerminal(a, r)

    -- Keep learning curve for λ = 0, 1 (doubles cannot be compared)
    if tostring(lambda) == '0' or tostring(lambda) == '1' then
      table.insert(epLambdaErrors, torch.sum(torch.pow((calcQ(theta) - QStar), 2)))
    end
  end

  -- Calculate error versus Q*
  local lambdaError = torch.sum(torch.pow((calcQ(theta) - QStar), 2))
  table.insert(lambdaErrors, lambdaError)

  -- Plot learning curve for λ = 0, 1 (doubles cannot be compared)
  if tostring(lambda) == '0' or tostring(lambda) == '1' then
    gnuplot.pngfigure('LinFuncApproxLambda' .. lambda .. 'Learning.png')
    gnuplot.plot('Sq. Error', torch.linspace(1, nEpisodes, nEpisodes), torch.Tensor(epLambdaErrors), '-')
    gnuplot.title('Sarsa(' .. lambda .. ') learning curve')
    gnuplot.ylabel('Squared error versus Q*')
    gnuplot.xlabel('Episode #')
    gnuplot.plotflush()
  end
end
-- Plot linear function approximator errors
gnuplot.pngfigure('LinFuncApproxLambda.png')
gnuplot.plot('Sq. Error', torch.linspace(0, 1, 11), torch.Tensor(lambdaErrors))
gnuplot.title('Linear function approximator errors')
gnuplot.ylabel('Squared error versus Q*')
gnuplot.xlabel('Lambda')
gnuplot.plotflush()

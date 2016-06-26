local nn = require 'nn'
local gnuplot = require 'gnuplot'
local environ = require 'environ'

local nEpisodes = 1000000
-- Number of discrete actions
local m = #environ.A

-- Initial exploration ɛ
local epsilon = 1
-- Linear ɛ decay
local epsilonDecay = 1/nEpisodes
-- Minimum ɛ
local epsilonMin = 0.05
-- Constant step-size ɑ
local alpha = 0.01

-- Create policy network π
local net = nn.Sequential()
net:add(nn.Linear(2, 32))
net:add(nn.ReLU(true))
net:add(nn.Linear(32, m))
net:add(nn.SoftMax())
-- Get network parameters θ
local theta, gradTheta = net:getParameters()

-- Results from each episode
local results = torch.Tensor(nEpisodes)

-- Sample
for i = 1, nEpisodes do
  -- Pick random starting state
  local s = {torch.random(1, 10), torch.random(1, 21)}
  
  -- Run till termination
  repeat
    -- Choose action by ɛ-greedy exploration
    local aIndex, probs
    if torch.uniform() < (1 - epsilon) then -- Act with probability 1 - ɛ
      -- Get categorical action distribution from π = p(s; θ)
      probs = net:forward(torch.Tensor(s))
      -- Sample action ~ p(s; θ)
      aIndex = torch.multinomial(probs, 1)[1]
    else
      probs = nil
      -- Otherwise pick any action with probability 1/m
      aIndex = torch.random(1, m)
    end
    local a = environ.A[aIndex]

    -- Perform a step
    local sPrime, r = environ.step(s, a) -- r comes from score function f(s)

    -- Use a policy gradient update (REINFORCE rule)
    -- ∇θ Es[f(s)] = ∇θ ∑s p(s)f(s) = Es[f(s) ∇θ logp(s)]
    if probs then
      -- Zero ∇θ
      gradTheta:zero()

      -- ∇θ logp(s) = 1/p(a) for chosen a, 0 otherwise
      local target = torch.zeros(m)
      target[m] = r * 1/probs[aIndex] -- f(s) ∇θ logp(s)
      
      -- Backpropagate
      net:backward(torch.Tensor(s), target)

      -- Gradient ascent (not descent)
      theta:add(alpha * gradTheta)
    end

    -- Set next state as current state
    s = sPrime

    -- Linearly decay ɛ
    epsilon = math.max(epsilon - epsilonDecay, epsilonMin)

    -- Save result of episode
    if environ.isTerminal(a, r) then
      results[i] = r
    end
  until environ.isTerminal(a, r)
end

-- Take average results over 1000 episodes
local avgResults = torch.Tensor(nEpisodes/1000)
for ep = 1, nEpisodes, 1000 do
  avgResults[(ep - 1)/1000 + 1] = torch.mean(results:narrow(1, ep, 1000))
end

-- Plot results
gnuplot.pngfigure('PolicyGradient.png')
gnuplot.plot('Average Result', torch.linspace(1, nEpisodes/1000, nEpisodes/1000), avgResults)
gnuplot.title('Policy Gradient Results')
gnuplot.ylabel('Result (Mean over 1000 Episodes)')
gnuplot.xlabel('Episode (x1000)')
gnuplot.plotflush()

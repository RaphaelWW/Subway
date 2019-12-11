include("./subway_simulator.jl")
using Random
using Flux
using Statistics: mean
using Zygote: gradient

function simulate(a_max, b_max, k, t_brake)
    return SubwaySimulator.simulate(a_max, b_max, k, t_brake)
end

Random.seed!(0)

model = Chain(Dense(4,16,relu), Dense(16,1,relu)) |> f64
θ = params(model)

function control(a_max, b_max, k, target)
    t_brake = model([a_max, b_max, k, target])
    return t_brake[1]
end

function distance_driven(a_max, b_max, k, target)
    t_brake = control(a_max, b_max, k, target)
    return simulate(a_max, b_max, k, t_brake)
end

function loss(a_max, b_max, k, target)
    return (distance_driven(a_max, b_max, k, target) - target)^2
end

function eval(losstype, dataset)
    sumloss = 0
    for data in dataset
        Flux.Optimise.update!(opt, θ, gradient(() -> loss(data...), θ))
        sumloss += loss(data...)
    end
    println(losstype,": ", sqrt(sumloss/length(dataset)))
end

K_INT = (10e-4,10e-3)
A_INT = (0.5,1.0)
DIST = (200.0,2000.0)

lerp(x, hi, lo) = x*(hi-lo)+lo
scenario() = (rand() * lerp(rand(),A_INT...), lerp(rand(),A_INT...), lerp(rand(),K_INT...), lerp(rand(),DIST...))

opt = ADAM(0.01,(0.9, 0.999))
trainset = [scenario() for i = 1:20_000]

#starting performance
eval("Pre-train loss", trainset)

#train
for epoch in range(1, stop=40)
    for data in trainset
        Flux.Optimise.update!(opt, θ, gradient(() -> loss(data...), θ))
    end
    eval(string("Train loss after epoch ", epoch), trainset)
end

#test
testset = [scenario() for i = 1:10_000]
eval("Test loss", testset)

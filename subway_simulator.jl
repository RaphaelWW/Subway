module SubwaySimulator
using DifferentialEquations
using Zygote: @adjoint
using Flux.Tracker: data

export simulate

function accelerate(ddu,du,u,p,t)
    a_max,k = p
    ddu[1] = -k*du[1]^2+a_max
end

function decelerate(ddu,du,u,p,t)
    b_max,k = p
    ddu[1] = -k*du[1]^2 - b_max
end

function simulate_acceleration(a_max, k, t_span)
    prob = SecondOrderODEProblem(accelerate, [0.0], [0.0], t_span, (a_max,k))
    return solve(prob, Nystrom4(), dt=0.1)
end

function simulate_acceleration_s(a_max, k, t_brake)
    return simulate_acceleration(a_max, k, (0.0, data(t_brake) + 1.0))(data(t_brake))[2]
end

function simulate_acceleration_v(a_max, k, t_brake)
    return simulate_acceleration(a_max, k, (0.0, data(t_brake) + 1.0))(data(t_brake))[1]
end

@adjoint function simulate_acceleration_s(a_max, k, t_brake)
    return simulate_acceleration_s(a_max, k, t_brake), ȳ -> (0, 0, ȳ*simulate_acceleration_v(a_max,k,t_brake))
end

@adjoint function simulate_acceleration_v(a_max, k, t_brake)
    v = simulate_acceleration_v(a_max, k, t_brake)
    return v, ȳ -> (0, 0, ȳ * (-k * v^2 + a_max))
end

function simulate_deceleration(b_max, k, v_max)
    t_span = (0.0, v_max/b_max + 1.0)
    prob = SecondOrderODEProblem(decelerate, [v_max], [0.0], t_span, (b_max,k))
    sol = solve(prob, Nystrom4(), dt = 0.1)

    i = 0.0
    while i < t_span[2] && sol(i)[1] > 0.0
        i += 0.1
    end

    return sol(i)[2]
end

@adjoint function simulate_deceleration(b_max, k, v_max)
    return simulate_deceleration(b_max, k, v_max), ȳ -> (0, 0, ȳ * v_max)
end

function simulate(a_max, b_max, k, t_brake)
    s1 = simulate_acceleration_s(a_max, k, t_brake)
    v_max = simulate_acceleration_v(a_max, k, t_brake)
    s2 = simulate_deceleration(b_max, k, v_max)
    return s1 + s2
end
end

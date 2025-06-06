using RigidityTheoryTools
using Documenter

DocMeta.setdocmeta!(RigidityTheoryTools, :DocTestSetup, :(using RigidityTheoryTools); recursive=true)

makedocs(;
    modules=[RigidityTheoryTools],
    authors="Sascha Stüttgen <sascha.stuettgen@rwth-aachen.de> and contributors",
    sitename="RigidityTheoryTools.jl",
    format=Documenter.HTML(;
        canonical="https://Saschobolt.github.io/RigidityTheoryTools.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Saschobolt/RigidityTheoryTools.jl",
    devbranch="main",
)

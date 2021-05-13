from setuptools import setup, find_packages

with open("README.md") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

setup(
    name="smoke",
    version="0.1.0",
    description="Smoke forecast model",
    long_description=readme,
    author=["Ashutosh Bhudia", "Brandon Dos Remedios", "Minnie Teng", "Ren Wang"],
    author_email=[
        "ashu.bhudia@gmail.com",
        "dosremedios.brandon@gmail.com",
        "m.tengy@gmail.com",
        "renwang435@gmail.com",
    ],
    entry_points={
        "console_scripts": [
            "frp-to-nc = smoke.convert.frp_to_nc:cli",
            "bluesky-to-nc = smoke.convert.bluesky_to_nc:cli",
            "firework-to-nc = smoke.convert.firework_to_nc:cli",
            "modis-aod-to-nc = smoke.convert.modis_aod_to_nc:cli",
            "smoke-plume-to-nc = smoke.convert.smoke_plume_to_nc:cli",
        ]
    },
    url="https://gitlab.math.ubc.ca/smoke/smoke",
    license=license,
    packages=find_packages(exclude=("tests", "docs")),
)

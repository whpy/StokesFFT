# StokesFFT
solving 2D stokes flow by cuda11
$\boldsymbol{f} = -\nabla p + \frac{1}{Re}\Delta\boldsymbol{u}$

##usage
the StokesFFT.cu in the folder StokesFFT/ is the most first version, while the file with structured suffix with overload of operator. But based on the baseline by comsol, the first version performs more physically. It is easy to compile and run, just enter the StokesFFT/ then 
```
make base
```
you would get a exe named SFFT then input
```
./SFFT 256
```
then the programme would calculate the Stokes equation in 2D periodic(four sides) with 256 grid points in both directions(x, y).
The programme would automatically generate the .csv files(u.csv, v.csv) to record the result, which could be visualized by matlab using the visualize.m file


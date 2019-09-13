# Mini wrapper for `dismod_at`

Wrapper(s) that fits a two-level nonlinear ODE-constrained random effects model using `dismod_at`.
Also includes code files for related utilities functions (e.g. plotting, matching measurement data with covariates).

### Install and Run `dismod_at`
* `dismod_at` can be installed locally via a docker image.
   Change the hash on the `git checkout` line in the Dockerfile to that of the
   [latest commit](https://github.com/bradbell/dismod_at/commits/master)
*  build a `dismod_at` image using
  ```docker build -t dismod_at:{tag} {path_to_docker_file}
  ```
*  use `docker images` to check that the image is built successfully.
*  to run the demo jupyter notebooks, use the command
   ```
   docker run -v {local_path_to_repo} -p 8888:8888 -it dismod_at:{tag}
   ```
   You can change the port to be different.

* inside a docker container, run
  ```
  jupyter notebook --generate-config
  jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
  ```
  Copy the link popped out in the terminal and open it in the browser.

### Wrapper files

* `two_level.py`: run a two_level nonlinear random effects model (i.e. one parent
   with multiple children). also has the option to run just one-level,
   ignoring the parent-children structure.
* `plot_two_level.py`: plot residuals, age/time pattern for rates and covariate values for each   child.
* `covariate.py`: match covariate to measurement data using population weighted smoothing.

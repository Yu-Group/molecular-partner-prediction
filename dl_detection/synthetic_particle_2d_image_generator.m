function [I_synth, I_object, I_psf, I_noise] = synthetic_particle_2d_image_generator(image_size, particle_prob, snr, sigma, varargin)
% The function is used for generate synthical images for particals through
% a microscope. The idea is to first sample particles under uniform
% distribution with a probabily, followed by applying PSF function. Then the
% amplitude based on the SNR (calculated over the Gauss noise) is applied. 
% At last, the Gauss and Poisson random noises are added to the image. 
%
%
% Inputs:    image_size : size of the synthetic image
%         particle_prob : the probability of particles occurred in the image
%                   snr : signal-to-noise ratio
%                 sigma : sigma for PSF, (sigma_xy, sigma_z
% 
% Options ('specifier', value):
%       'psf_kernel_size' : kernel size for psf (Gauss kernel)
%       'gauss_noise_amp' : amptitude for gauss noise (mean value)
%     'gauss_noise_sigma' : sigma for gauss noise
%     'poisson_noise_amp' : amptitude for poisson noise
%  'poisson_noise_lambda' : lambda parameter for poisson noise
%             'rand_seed' : seed to control the random generator
% 
% OUtput:    I_synth : synthetic image with rand noise
%            I_objec : image showing actual partical locations
%              I_psf : image after apply psf (without amptitude)
%            I_noise : noise image for both Gauss and Poisson noises
% 
% Xiongtao Ruan, Augest 2019
% 
% 08/09/2019 add different modes for psf patterns and noise types


if nargin < 1
    image_size = [2048, 2048];
    particle_prob = 0.0001;
    snr = 3;
    sigma = 1.26;
end

ip = inputParser;
ip.CaseSensitive = false;
ip.addRequired('image_size', @isnumeric);
ip.addRequired('particle_prob', @isnumeric);
ip.addRequired('snr', @isnumeric);
ip.addRequired('sigma', @isnumeric)

ip.addParameter('psf_kernel_size', [9, 9], @isnumeric);
ip.addParameter('gauss_noise_amp', 100, @isnumeric);
ip.addParameter('gauss_noise_sigma', 4, @isnumeric);
ip.addParameter('poisson_noise_amp', 12, @isnumeric);
ip.addParameter('poisson_noise_lambda', 0.001, @isnumeric);
ip.addParameter('rand_seed', 1, @isnumeric);

ip.parse(image_size, particle_prob, snr, sigma, varargin{:});


image_size = ip.Results.image_size;
particle_prob = ip.Results.particle_prob;
snr = ip.Results.snr;
sigma = ip.Results.sigma;
psf_kernel_size = ip.Results.psf_kernel_size;
gauss_noise_amp = ip.Results.gauss_noise_amp;
gauss_noise_sigma = ip.Results.gauss_noise_sigma;
poisson_noise_amp = ip.Results.poisson_noise_amp;
poisson_noise_lambda = ip.Results.poisson_noise_lambda;
rand_seed = ip.Results.rand_seed;

rng(rand_seed);

% generate particles
I_object = rand(image_size) < particle_prob;

psf_gauss_kernel = fspecial('Gaussian', psf_kernel_size, sigma(1));
normal_factor = psf_gauss_kernel(floor(psf_kernel_size(1) + 1) / 2, floor(psf_kernel_size(1) + 1) / 2);
I_psf = imfilter(double(I_object), psf_gauss_kernel / normal_factor, 'conv', 'same', 0);

signal_A = gauss_noise_sigma * snr;
I_psf_A = signal_A * I_psf;

% gaussian noise
I_gauss_noise = gauss_noise_amp + randn(image_size) * gauss_noise_sigma;

% poisson noise
I_poisson_noise = poissrnd(poisson_noise_lambda, image_size) * poisson_noise_amp;

I_noise = I_gauss_noise + I_poisson_noise;

I_synth = I_psf_A + I_noise;
I_synth = uint16(I_synth);

end


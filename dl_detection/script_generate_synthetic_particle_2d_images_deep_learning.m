% Generate images for particles with different settings
% generate for different psfs: Gaussian with different sigmas, Laplacian
% and customized with two separate peaks. 
% Here we keep the same total laser power. 

% generate synthetic images for deep learning
clear, close all

image_directory = '/scratch/users/vision/data/abc_data/synthetic_2d/';
% image_directory = '../results/synthetic_particle_2d_images_dl_09202019/';
mkdir(image_directory);
addpath('utils');
s = RandStream('mt19937ar','Seed',0);
RandStream.setGlobalStream(s);

snr_mat = 0.5 : 0.25 : 5;
snr_mat = [1.5, 2, 3];
% rand_seed_mat = [1, 48, 652, 1234, 21083, 907845, 4627528, 38389335, 93047818, 1239012832];
particle_prob_mat = [5e-5, 1e-4, 1e-3];
psf_method_names = {'gaussian'};
psf_method_set = 1;
psf_method_param = 1.26;

image_per_group = 10000;

image_size = [512, 512];
particle_prob = 0.001;
snr = 15;
sigma = 1.26;
gauss_noise_amp = 100;
gauss_noise_sigma = 4;
poisson_noise_amp = 12;
poisson_noise_lambda = 0.001;
% poisson_noise_lambda = 0.000;
kernel_size = 9;

standard_gaussian_kernel = fspecial('gaussian', kernel_size, sigma);
standard_gaussian_kernel = standard_gaussian_kernel / standard_gaussian_kernel(floor((kernel_size + 1) / 2), floor((kernel_size + 1) / 2));

kernel_factor = sum(standard_gaussian_kernel(:));

for prob_ind = 1 : numel(particle_prob_mat)
    particle_prob = particle_prob_mat(prob_ind);
    rand_seed_mat = sort(randi(2^32 - 1, image_per_group, 1));
    for i = 1 : numel(rand_seed_mat)
        seed_i = rand_seed_mat(i);    
        [I_synth, I_object, I_psf, I_noise] = synthetic_particle_2d_image_generator(image_size, particle_prob, snr, sigma, ...
                                               'gauss_noise_amp', gauss_noise_amp, 'gauss_noise_sigma', gauss_noise_sigma, ...
                                               'poisson_noise_amp', poisson_noise_amp, 'poisson_noise_lambda', poisson_noise_lambda, ...
                                               'rand_seed', seed_i);
        
        for psf_ind = 1 : numel(psf_method_set) 
            cur_psf_id = psf_method_set(psf_ind);
            cur_psf_name = psf_method_names{cur_psf_id};
            cur_psf_param = psf_method_param(psf_ind);
            
            switch cur_psf_name
                case 'gaussian'
                    I_kernel = fspecial('gaussian', kernel_size, cur_psf_param);
                case 'laplacian'
                    I_kernel = lapacian_distribution_2d_kernel(kernel_size, cur_psf_param);                  
            end
            I_kernel = I_kernel * kernel_factor;
            I_conv = imfilter(double(I_object), I_kernel, 'conv', 'same', 0);
            
            img_path = sprintf('%sgt_randseed_%d_particleprob_%g.tiff', ...
                image_directory, seed_i, particle_prob);
            writetiff(uint8(I_object), img_path)
            for k = 1 : numel(snr_mat)
                snr = snr_mat(k);
                I_syn = I_conv * (gauss_noise_sigma * snr) + I_noise;

                img_path = sprintf('%ssynthetic_kernel_%s_param_%g_randseed_%d_particleprob_%g_snr_%g.tiff', ...
                            image_directory, cur_psf_name, cur_psf_param, seed_i, particle_prob, snr)
                writetiff(uint8(I_syn), img_path);
            end
        end
    end
end


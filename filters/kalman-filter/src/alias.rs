use nalgebra as na;

pub type State<const C: usize> = na::SVector<f64, C>;
pub type Covariance<const C: usize> = na::SMatrix<f64, C, C>;

pub type CurrentState<const C: usize> = (State<C>, Covariance<C>);

pub type TransitionModel<const C: usize> = na::SMatrix<f64, C, C>;
pub type TransitionNoise<const C: usize> = na::SMatrix<f64, C, C>;

pub type Measurement<const R: usize> = na::SVector<f64, R>;
pub type MeasurementModel<const R: usize, const C: usize> = na::SMatrix<f64, R, C>;
pub type MeasurementNoise<const R: usize> = na::SMatrix<f64, R, R>;

pub type Observation<const R: usize> = (Measurement<R>, MeasurementNoise<R>);

pub type InputModel<const R: usize, const C: usize> = na::SMatrix<f64, R, C>;
pub type InputVector<const R: usize> = na::SVector<f64, R>;

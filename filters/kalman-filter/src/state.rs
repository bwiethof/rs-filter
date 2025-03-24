use nalgebra as na;
use crate::alias::CurrentState;

pub struct State<const C: usize> {
    estimate: na::SVector<f64, C>,
    covariance: na::SMatrix<f64, C, C>,
}

impl<const C: usize> State<C> {
    fn new(initial:  CurrentState<C>) -> Self {
        State {
            estimate: initial.0,
            covariance: initial.1,
        }
    }
}

impl<const C: usize> Default for State<C> {
    fn default() -> Self {
        State {
            estimate: na::SVector::zeros(),
            covariance: na::SMatrix::identity(),
        }
    }
}

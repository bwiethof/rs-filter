use crate::alias::*;
use nalgebra as na;
use nalgebra::SMatrix;
use std::fmt::Debug;

#[derive(Debug)]
pub struct TransitionError {}

struct KalmanFilter<const TC: usize = 1, const MR: usize = 1, const BR: usize = 1> {
    // Model might be better somewhere else separate -> interface with model might be better but there hsould be more constrainst what the model should look like
    transition_model: TransitionModel<TC>,
    measurement_model: MeasurementModel<MR, TC>,
    measurement_noise: TransitionNoise<TC>,
    state: State<TC>,
    covariance: Covariance<TC>,
    input_model: InputModel<TC, BR>,
}

impl<const TC: usize, const MR: usize, const BR: usize> Default for KalmanFilter<TC, MR, BR> {
    fn default() -> Self {
        KalmanFilter {
            transition_model: TransitionModel::<TC>::identity(),
            state: na::zero::<State<TC>>(),
            covariance: na::zero::<Covariance<TC>>(),
            measurement_model: na::zero::<MeasurementModel<MR, TC>>(),
            measurement_noise: TransitionNoise::<TC>::identity(),
            input_model: InputModel::<TC, BR>::zeros(),
        }
    }
}

impl<const TC: usize, const MR: usize, const BR: usize> KalmanFilter<TC, MR, BR> {
    pub fn new(
        transition_model: TransitionModel<TC>,
        measurement_model: MeasurementModel<MR, TC>,
        measurement_noise: MeasurementNoise<TC>,
    ) -> Self {
        KalmanFilter {
            transition_model,
            measurement_model,
            measurement_noise,
            state: na::zero(),
            covariance: Covariance::<TC>::identity(),
            input_model: InputModel::<TC, BR>::zeros(),
        }
    }

    pub fn with_state(mut self, state: State<TC>, covariance: Covariance<TC>) -> Self {
        self.state = state;
        self.covariance = covariance;
        self
    }

    pub fn with_input_model(mut self, model: InputModel<TC, BR>) -> Self {
        self.input_model = model;
        self
    }

    pub fn step(
        &mut self,
        dt: f64,
        observation: Observation<MR>,
        u: Option<InputVector<BR>>,
    ) -> Result<CurrentState<TC>, TransitionError> {
        // a-priori
        let predicted = self.predict((self.state, self.covariance), dt, u)?;

        // a-posteriori
        let (state, cov) = self.update(predicted, observation)?;
        
        // Update internal representation
        self.state = state;
        self.covariance = cov;

        Ok((state, cov))
    }

    fn predict(
        &self,
        (mut state, mut covariance): CurrentState<TC>,
        dt: f64,
        u: Option<InputVector<BR>>,
    ) -> Result<CurrentState<TC>, TransitionError> {
        if dt <= 0.0 {
            return Err(TransitionError {});
        }

        let transition = self.transition_model * dt;

        state += transition * state + self.input_model * u.unwrap_or(InputVector::<BR>::zeros());

        covariance += transition * covariance * transition.transpose() + self.measurement_noise;

        Ok((state, covariance))
    }

    fn update(
        &self,
        (state, covariance): CurrentState<TC>,
        (measurement, noise): Observation<MR>,
    ) -> Result<CurrentState<TC>, TransitionError> {
        // Calculate innovation
        let innovation = measurement - self.measurement_model * state;
        let innovation_matrix =
            self.measurement_model * covariance * self.measurement_model.transpose() + noise;

        // Calculate gain if possible
        let gain = match innovation_matrix.try_inverse() {
            None => Err(TransitionError {}),
            Some(inv) => Ok(covariance * self.measurement_model.transpose() * inv),
        }?;

        // Update the actual state
        let new_state = state + gain * innovation;
        let new_covariance: Covariance<TC> =
            (Covariance::<TC>::identity() - gain * self.measurement_model) * covariance;

        Ok((new_state, new_covariance))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector2;

    #[test]
    fn predict_state() -> Result<(), TransitionError> {
        let transition_model = TransitionModel::<1>::identity() * 2f64;
        let measurement_model = MeasurementModel::<1, 1>::identity();
        let measurement_noise = TransitionNoise::<1>::identity() * 2f64;

        let initial_state = State::<1>::new(1.0);
        let initial_covariance = Covariance::<1>::identity();

        let filter: KalmanFilter =
            KalmanFilter::new(transition_model, measurement_model, measurement_noise);
        let pred_result = filter.predict((initial_state, initial_covariance), 1.0, None);

        assert!(pred_result.is_ok());
        if let Ok((state, cov)) = pred_result {
            assert_eq!(state, State::<1>::new(2.0));
            assert_eq!(cov, Covariance::<1>::identity() * 6.0);
        }

        // wrong input shall return an error
        let pred_result = filter.predict((initial_state, initial_covariance), -1.0, None);
        assert!(pred_result.is_err());

        Ok(())
    }

    #[test]
    fn with_input_model() -> Result<(), TransitionError> {
        let transition_model = TransitionModel::<1>::identity() * 2f64;
        let measurement_model = MeasurementModel::<1, 1>::identity();
        let measurement_noise = TransitionNoise::<1>::identity() * 2f64;

        let initial_state = State::<1>::new(1.0);
        let initial_covariance = Covariance::<1>::identity();

        let filter: KalmanFilter<1, 1, 2> =
            KalmanFilter::new(transition_model, measurement_model, measurement_noise)
                .with_input_model(InputModel::<1, 2>::new(1.0, 2.0));
        let (state, cov) = filter.predict((initial_state, initial_covariance), 1.0, None)?;

        assert_eq!(state, State::<1>::new(2.0));
        assert_eq!(cov, Covariance::<1>::identity() * 6.0);

        let (state, cov) = filter.predict(
            (initial_state, initial_covariance),
            1.0,
            Some(InputVector::<2>::new(1.0, 1.0)),
        )?;

        assert_eq!(state, State::<1>::new(5.0));
        assert_eq!(cov, Covariance::<1>::identity() * 6.0);

        Ok(())
    }

    #[test]
    fn update_state() -> Result<(), TransitionError> {
        let transition_model = TransitionModel::<2>::identity() * 3f64;
        let measurement_model = MeasurementModel::<1, 2>::new(1.0, 1.0);
        let transition_noise = TransitionNoise::<2>::identity() * 2f64;

        let initial_state = State::<2>::new(1.0, 2.0);
        let initial_covariance = Covariance::<2>::identity();

        let observation: Observation<1> = (
            Measurement::<1>::new(2.0),
            MeasurementNoise::<1>::identity() * 2f64,
        );

        let filter: KalmanFilter<2, 1> =
            KalmanFilter::new(transition_model, measurement_model, transition_noise);

        let pred_result = filter.update((initial_state, initial_covariance), observation);
        assert!(pred_result.is_ok());
        if let Ok((state, cov)) = pred_result {
            let eig_values = cov.eigenvalues();
            assert!(eig_values.is_some());
            let eig_values = eig_values.unwrap();
            assert!(eig_values.sum() > 0.0);

            assert_eq!(state, State::<2>::new(0.75, 1.75));
            assert_eq!(
                cov,
                Covariance::<2>::from_columns(&[
                    Vector2::new(0.75, -0.25),
                    Vector2::new(-0.25, 0.75)
                ])
            );
        }

        Ok(())
    }
}

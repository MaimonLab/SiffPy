#ifndef EXP_MATH_HPP
#define EXP_MATH_HPP

#include <math.h>
#include <Python.h>
#include <vector>


double mono_exp(uint16_t x, double tau, double sigma, double tauo) {
    // TODO: NOT QUITE RIGHT, USES EDGE INSTEAD OF INTEGRATING BIN
    double_t gauss_coeff = (1.0/(2.0*tau)) * exp(pow(sigma,2.0)/(2.0*pow(tau,2.0)));
    double_t normalization = erfc( -(tau * (x-tauo) - pow(sigma,2.0))/ (sqrt(2.0)*sigma*tau) );
    double_t exp_dist = exp(-(x-tauo)/tau);

    if (x > tauo) return gauss_coeff * normalization * exp_dist;
    else return 0;
}

std::vector<double_t> compute_arrival_p(PyObject* FLIMParams){
    PyObject* ncomp = PyObject_GetAttrString(FLIMParams, "ncomponents");
    uint16_t n_components = PyLong_AS_LONG(ncomp);
    Py_DECREF(ncomp);

    PyObject* exps = PyObject_GetAttrString(FLIMParams, "exps"); // list of Exp objects,
    uint16_t n_bins = 1024; // hardcoded for now, TODO fix

    std::vector<double_t> arrival_p(n_bins,0); // initialize as 0s

    PyObject* T_O = PyObject_GetAttrString(FLIMParams, "T_O");
    double_t tauo = PyFloat_AS_DOUBLE(T_O);
    Py_DECREF(T_O);

    PyObject* tau_g = PyObject_GetAttrString(FLIMParams, "tau_g");
    double_t sigma = PyFloat_AS_DOUBLE(tau_g);
    Py_DECREF(tau_g);    

    //throw std::runtime_error("Am I here?");
    // compute monoexponential probs times fraction
    for(uint16_t compIdx = 0; compIdx < n_components; compIdx++) {
        PyObject* component_params = PyList_GetItem(exps, Py_ssize_t(compIdx)); // a dict, borrowed ref

        double_t frac = PyFloat_AS_DOUBLE(PyObject_GetAttrString(component_params, "frac")); // borrowed ref
        double_t tau = PyFloat_AS_DOUBLE(PyObject_GetAttrString(component_params, "tau")); // borrowed ref

        for(uint16_t x = 0; x<n_bins; x++) {
            arrival_p[x] += frac*mono_exp(x,tau,sigma,tauo);
        }
    }
    Py_DECREF(exps);
    return arrival_p;
};

#endif
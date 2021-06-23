#ifndef EXP_MATH_HPP
#define EXP_MATH_HPP

#include <math.h>
#include <Python.h>
#include <vector>


double mono_exp(uint16_t x, double tau, double sigma, double tauo) {
    // TODO: NOT QUITE RIGHT, USES EDGE INSTEAD OF INTEGRATING BIN
    double_t gauss_coeff = (1.0/(2.0*tau)) * exp(pow(sigma,2.0)/(2.0*pow(tau,2.0)));
    double_t normalization = erfc( -(tau * (x-tauo) + pow(sigma,2.0))/ (sqrt(2.0)*sigma*tau) );
    double_t exp_dist = exp(-(x-tauo)/tau);

    if (x > tauo) return gauss_coeff * normalization * exp_dist;
    else return 0;
}

std::vector<double_t> compute_arrival_p(PyObject* FLIMParams){
    PyObject* ncomp = PyObject_GetAttrString(FLIMParams, "Ncomponents");
    uint16_t n_components = PyLong_AS_LONG(ncomp);
    Py_DECREF(ncomp);

    PyObject* exp_params = PyObject_GetAttrString(FLIMParams, "Exp_params"); // list of dicts,

    uint16_t n_bins = 1024; // hardcoded for now, TODO fix

    std::vector<double_t> arrival_p(n_bins,0); // initialize as 0s

    PyObject* T_O = PyObject_GetAttrString(FLIMParams, "T_O");
    double_t tauo = PyFloat_AS_DOUBLE(T_O);
    Py_DECREF(T_O);

    PyObject* IRF = PyObject_GetAttrString(FLIMParams, "IRF");
    double_t sigma = PyFloat_AS_DOUBLE(PyDict_GetItemString(PyDict_GetItemString(IRF,"PARAMS"),"SIGMA"));
    Py_DECREF(IRF);    

    // compute monoexponential probs times fraction
    for(uint16_t compIdx = 0; compIdx < n_components; compIdx++) {
        PyObject* component_params = PyList_GetItem(exp_params, Py_ssize_t(compIdx)); // a dict, borrowed ref

        double_t frac = PyFloat_AS_DOUBLE(PyDict_GetItemString(component_params, "FRAC")); // borrowed ref
        double_t tau = PyFloat_AS_DOUBLE(PyDict_GetItemString(component_params, "TAU")); // borrowed ref

        for(uint16_t x = 0; x<n_bins; x++) {
            arrival_p[x] += frac*mono_exp(x,tau,sigma,tauo);
        }
    }
    Py_DECREF(exp_params);
    return arrival_p;
};

#endif
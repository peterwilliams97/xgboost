#ifndef XGBOOST_PYTHON_H
#define XGBOOST_PYTHON_H

#ifdef WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif


/*!
 * \file xgboost_python.h
 * \author Tianqi Chen
 * \brief python wrapper for xgboost, using ctypes,
 *        hides everything behind functions
 *      use c style interface
 */
#include "../booster/xgboost_data.h"



extern "C"{
    /*! \brief type of row entry */
    typedef xgboost::booster::FMatrixS::REntry XGEntry;
    
    /*! 
     * \brief create a data matrix 
     * \return a new data matrix
     */
    DLLEXPORT void *XGDMatrixCreate(void);
    /*! 
     * \brief free space in data matrix
     */
    DLLEXPORT void XGDMatrixFree(void *handle);
    /*! 
     * \brief load a data matrix from text file or buffer(if exists)
     * \param handle a instance of data matrix
     * \param fname file name 
     * \param silent print statistics when loading
     */
    DLLEXPORT void XGDMatrixLoad(void *handle, const char *fname, int silent);
    /*!
     * \brief load a data matrix into binary file
     * \param handle a instance of data matrix
     * \param fname file name 
     * \param silent print statistics when saving
     */
    DLLEXPORT void XGDMatrixSaveBinary(void *handle, const char *fname, int silent);
    /*! 
     * \brief set matrix content from csr format
     * \param handle a instance of data matrix
     * \param indptr pointer to row headers
     * \param indices findex
     * \param data    fvalue
     * \param nindptr number of rows in the matix + 1 
     * \param nelem number of nonzero elements in the matrix
     */
    DLLEXPORT void XGDMatrixParseCSR( void *handle,
                            const size_t *indptr,
                            const unsigned *indices,
                            const float *data,
                            size_t nindptr,
                            size_t nelem );
    /*! 
     * \brief set matrix content from data content
     * \param handle a instance of data matrix
     * \param data pointer to the data space
     * \param nrow number of rows
     * \param ncol number columns
     * \param missing which value to represent missing value
     */
    DLLEXPORT void XGDMatrixParseMat( void *handle,
                            const float *data,
                            size_t nrow,
                            size_t ncol,
                            float  missing );
    /*! 
     * \brief set label of the training matrix
     * \param handle a instance of data matrix
     * \param label pointer to label
     * \param len length of array
     */    
    DLLEXPORT void XGDMatrixSetLabel( void *handle, const float *label, size_t len );
    /*! 
     * \brief set label of the training matrix
     * \param handle a instance of data matrix
     * \param group pointer to group size
     * \param len length of array
     */    
    DLLEXPORT void XGDMatrixSetGroup( void *handle, const unsigned *group, size_t len );
    /*! 
     * \brief set weight of each instacne
     * \param handle a instance of data matrix
     * \param weight data pointer to weights
     * \param len length of array
     */    
    DLLEXPORT void XGDMatrixSetWeight( void *handle, const float *weight, size_t len );
    /*! 
     * \brief get label set from matrix
     * \param handle a instance of data matrix
     * \param len used to set result length
     * \return pointer to the label
     */
    DLLEXPORT const float *XGDMatrixGetLabel( const void *handle, size_t* len );
    /*! 
     * \brief get weight set from matrix
     * \param handle a instance of data matrix
     * \param len used to set result length
     * \return pointer to the weight
     */
    DLLEXPORT const float *XGDMatrixGetWeight( const void *handle, size_t* len );
    /*! 
     * \brief clear all the records, including feature matrix and label
     * \param handle a instance of data matrix
     */
    DLLEXPORT void XGDMatrixClear(void *handle);
    /*! 
     * \brief return number of rows
     */    
    DLLEXPORT size_t XGDMatrixNumRow(const void *handle);
    /*! 
     * \brief add row 
     * \param handle a instance of data matrix
     * \param data array of row content 
     * \param len length of array
     */
    DLLEXPORT void XGDMatrixAddRow(void *handle, const XGEntry *data, size_t len);
    /*! 
     * \brief get ridx-th row of sparse matrix
     * \param handle handle
     * \param ridx row index 
     * \param len used to set result length
     * \reurn pointer to the row
     */    
    DLLEXPORT const XGEntry *XGDMatrixGetRow(void *handle, unsigned ridx, size_t* len);
    
    // --- start XGBoost class
    /*! 
     * \brief create xgboost learner 
     * \param dmats matrices that are set to be cached
     * \param create a booster
     */
    DLLEXPORT void *XGBoosterCreate( void* dmats[], size_t len ); 
    /*! 
     * \brief free obj in handle 
     * \param handle handle to be freed
     */
    DLLEXPORT void XGBoosterFree( void* handle ); 
    /*! 
     * \brief set parameters 
     * \param handle handle
     * \param name  parameter name
     * \param val value of parameter
     */    
    DLLEXPORT void XGBoosterSetParam( void *handle, const char *name, const char *value );   
    /*! 
     * \brief update the model in one round using dtrain
     * \param handle handle
     * \param dtrain training data
     */        
    DLLEXPORT void XGBoosterUpdateOneIter( void *handle, void *dtrain );   
    
    /*!
     * \brief update the model, by directly specify gradient and second order gradient, 
     *        this can be used to replace UpdateOneIter, to support customized loss function
     * \param handle handle
     * \param dtrain training data
     * \param grad gradient statistics
     * \param hess second order gradient statistics
     * \param len length of grad/hess array
     * \param bst_group boost group we are working at, default = -1
     */
    DLLEXPORT void XGBoosterBoostOneIter( void *handle, void *dtrain, 
                                float *grad, float *hess, size_t len, int bst_group );   
    /*! 
     * \brief print evaluation statistics to stdout for xgboost
     * \param handle handle
     * \param iter current iteration rounds
     * \param dmats pointers to data to be evaluated
     * \param evnames pointers to names of each data
     * \param len  length of dmats
     */ 
    DLLEXPORT void XGBoosterEvalOneIter( void *handle, int iter, void *dmats[], const char *evnames[], size_t len ); 
    /*! 
     * \brief make prediction based on dmat
     * \param handle handle
     * \param dmat data matrix
     * \param len used to store length of returning result
     * \param bst_group booster group, if model contains multiple booster group, default = -1 means predict for all groups
     */
    DLLEXPORT const float *XGBoosterPredict( void *handle, void *dmat, size_t *len, int bst_group );
    /*! 
     * \brief load model from existing file
     * \param handle handle
     * \param fname file name
     */
    DLLEXPORT void XGBoosterLoadModel( void *handle, const char *fname );
    /*! 
     * \brief save model into existing file
     * \param handle handle
     * \param fname file name
     */
    DLLEXPORT void XGBoosterSaveModel( const void *handle, const char *fname );
    /*! 
     * \brief dump model into text file
     * \param handle handle
     * \param fname file name
     * \param fmap  name to fmap can be empty string
     */ 
    DLLEXPORT void XGBoosterDumpModel( void *handle, const char *fname, const char *fmap );
    /*! 
     * \brief interactively update model: beta
     * \param handle handle
     * \param dtrain training data
     * \param action action name
     */
    DLLEXPORT void XGBoosterUpdateInteract( void *handle, void *dtrain, const char* action );
};
#endif


package com.iss_mr_grp7_job_recommender.iss_mr_grp7_job_recommender;

/**
 * This class was automatically generated by the data modeler tool.
 */

@org.kie.api.definition.type.Label("Validation Error")
public class ValidationErrorDO implements java.io.Serializable {

	static final long serialVersionUID = 1L;

	@org.kie.api.definition.type.Label("Error description")
	private java.lang.String error;

	public ValidationErrorDO() {
	}

	public java.lang.String getError() {
		return this.error;
	}

	public void setError(java.lang.String error) {
		this.error = error;
	}

	public ValidationErrorDO(java.lang.String error) {
		this.error = error;
	}

}
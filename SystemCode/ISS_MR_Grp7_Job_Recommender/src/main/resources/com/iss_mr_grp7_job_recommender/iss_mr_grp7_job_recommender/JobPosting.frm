{"id":"f9af0878-6b41-41d9-9307-1754760cd69a","name":"JobPosting","model":{"source":"INTERNAL","className":"com.iss_mr_grp7_job_recommender.iss_mr_grp7_job_recommender.JobPosting","name":"JobPosting","properties":[{"name":"companyName","typeInfo":{"type":"BASE","className":"java.lang.String","multiple":false},"metaData":{"entries":[{"name":"field-label","value":"Company Name"},{"name":"field-placeHolder","value":"Company Name"}]}},{"name":"jobTitle","typeInfo":{"type":"BASE","className":"java.lang.String","multiple":false},"metaData":{"entries":[{"name":"field-label","value":"Job Title"},{"name":"field-placeHolder","value":"Job Title"}]}},{"name":"companyLocation","typeInfo":{"type":"BASE","className":"java.lang.String","multiple":false},"metaData":{"entries":[{"name":"field-label","value":"Company Location"},{"name":"field-placeHolder","value":"Company Location"}]}},{"name":"jobRequirements","typeInfo":{"type":"BASE","className":"java.lang.String","multiple":false},"metaData":{"entries":[{"name":"field-label","value":"Job Requirements"},{"name":"field-placeHolder","value":"Job Requirements"}]}},{"name":"employmentType","typeInfo":{"type":"BASE","className":"java.lang.String","multiple":false},"metaData":{"entries":[{"name":"field-label","value":"Employment Type"},{"name":"field-placeHolder","value":"Employment Type"}]}},{"name":"minSalary","typeInfo":{"type":"BASE","className":"java.lang.Integer","multiple":false},"metaData":{"entries":[{"name":"field-label","value":"Minimum Salary"},{"name":"field-placeHolder","value":"Minimum Salary"}]}},{"name":"maxSalary","typeInfo":{"type":"BASE","className":"java.lang.Integer","multiple":false},"metaData":{"entries":[{"name":"field-label","value":"Maximum Salary"},{"name":"field-placeHolder","value":"Maximum Salary"}]}},{"name":"seniorityLevel","typeInfo":{"type":"BASE","className":"java.lang.String","multiple":false},"metaData":{"entries":[{"name":"field-label","value":"Seniority Level"},{"name":"field-placeHolder","value":"Seniority Level"}]}},{"name":"jobCategory","typeInfo":{"type":"BASE","className":"java.lang.String","multiple":false},"metaData":{"entries":[{"name":"field-label","value":"Job Category"},{"name":"field-placeHolder","value":"Job Category"}]}},{"name":"workExperienceReq","typeInfo":{"type":"BASE","className":"java.lang.Integer","multiple":false},"metaData":{"entries":[{"name":"field-label","value":"Years of Work Experience Required"},{"name":"field-placeHolder","value":"Years of Work Experience Required"}]}}],"formModelType":"org.kie.workbench.common.forms.data.modeller.model.DataObjectFormModel"},"fields":[{"maxLength":100,"placeHolder":"Employment Type","id":"field_1876","name":"employmentType","label":"Employment Type","required":false,"readOnly":false,"validateOnChange":true,"helpMessage":"","binding":"employmentType","standaloneClassName":"java.lang.String","code":"TextBox","serializedFieldClassName":"org.kie.workbench.common.forms.fields.shared.fieldTypes.basic.textBox.definition.TextBoxFieldDefinition"},{"maxLength":100,"placeHolder":"Company Name","id":"field_0194","name":"companyName","label":"Company Name","required":false,"readOnly":false,"validateOnChange":true,"helpMessage":"","binding":"companyName","standaloneClassName":"java.lang.String","code":"TextBox","serializedFieldClassName":"org.kie.workbench.common.forms.fields.shared.fieldTypes.basic.textBox.definition.TextBoxFieldDefinition"},{"maxLength":100,"placeHolder":"Job Title","id":"field_51521","name":"jobTitle","label":"Job Title","required":false,"readOnly":false,"validateOnChange":true,"helpMessage":"","binding":"jobTitle","standaloneClassName":"java.lang.String","code":"TextBox","serializedFieldClassName":"org.kie.workbench.common.forms.fields.shared.fieldTypes.basic.textBox.definition.TextBoxFieldDefinition"},{"placeHolder":"Years of Work Experience Required","maxLength":100,"id":"field_235","name":"workExperienceReq","label":"Years of Work Experience Required","required":false,"readOnly":false,"validateOnChange":true,"helpMessage":"","binding":"workExperienceReq","standaloneClassName":"java.lang.Integer","code":"IntegerBox","serializedFieldClassName":"org.kie.workbench.common.forms.fields.shared.fieldTypes.basic.integerBox.definition.IntegerBoxFieldDefinition"},{"maxLength":100,"placeHolder":"Seniority Level","id":"field_4012","name":"seniorityLevel","label":"Seniority Level","required":false,"readOnly":false,"validateOnChange":true,"helpMessage":"","binding":"seniorityLevel","standaloneClassName":"java.lang.String","code":"TextBox","serializedFieldClassName":"org.kie.workbench.common.forms.fields.shared.fieldTypes.basic.textBox.definition.TextBoxFieldDefinition"},{"maxLength":100,"placeHolder":"Job Category","id":"field_4314","name":"jobCategory","label":"Job Category","required":false,"readOnly":false,"validateOnChange":true,"helpMessage":"","binding":"jobCategory","standaloneClassName":"java.lang.String","code":"TextBox","serializedFieldClassName":"org.kie.workbench.common.forms.fields.shared.fieldTypes.basic.textBox.definition.TextBoxFieldDefinition"},{"maxLength":100,"placeHolder":"Job Requirements","id":"field_657","name":"jobRequirements","label":"Job Requirements","required":false,"readOnly":false,"validateOnChange":true,"helpMessage":"","binding":"jobRequirements","standaloneClassName":"java.lang.String","code":"TextBox","serializedFieldClassName":"org.kie.workbench.common.forms.fields.shared.fieldTypes.basic.textBox.definition.TextBoxFieldDefinition"},{"placeHolder":"Minimum Salary","maxLength":100,"id":"field_7139","name":"minSalary","label":"Minimum Salary","required":false,"readOnly":false,"validateOnChange":true,"helpMessage":"","binding":"minSalary","standaloneClassName":"java.lang.Integer","code":"IntegerBox","serializedFieldClassName":"org.kie.workbench.common.forms.fields.shared.fieldTypes.basic.integerBox.definition.IntegerBoxFieldDefinition"},{"placeHolder":"Maximum Salary","maxLength":100,"id":"field_9469","name":"maxSalary","label":"Maximum Salary","required":false,"readOnly":false,"validateOnChange":true,"helpMessage":"","binding":"maxSalary","standaloneClassName":"java.lang.Integer","code":"IntegerBox","serializedFieldClassName":"org.kie.workbench.common.forms.fields.shared.fieldTypes.basic.integerBox.definition.IntegerBoxFieldDefinition"}],"layoutTemplate":{"version":2,"style":"FLUID","layoutProperties":{},"rows":[{"height":"12","properties":{},"layoutColumns":[{"span":"12","height":"12","properties":{},"rows":[],"layoutComponents":[{"dragTypeName":"org.kie.workbench.common.forms.editor.client.editor.rendering.EditorFieldLayoutComponent","properties":{"field_id":"field_1876","form_id":"f9af0878-6b41-41d9-9307-1754760cd69a"}}]}]},{"height":"12","properties":{},"layoutColumns":[{"span":"12","height":"12","properties":{},"rows":[],"layoutComponents":[{"dragTypeName":"org.kie.workbench.common.forms.editor.client.editor.rendering.EditorFieldLayoutComponent","properties":{"field_id":"field_51521","form_id":"f9af0878-6b41-41d9-9307-1754760cd69a"}}]}]},{"height":"12","properties":{},"layoutColumns":[{"span":"12","height":"12","properties":{},"rows":[],"layoutComponents":[{"dragTypeName":"org.kie.workbench.common.forms.editor.client.editor.rendering.EditorFieldLayoutComponent","properties":{"field_id":"field_0194","form_id":"f9af0878-6b41-41d9-9307-1754760cd69a"}}]}]},{"height":"12","properties":{},"layoutColumns":[{"span":"12","height":"12","properties":{},"rows":[],"layoutComponents":[{"dragTypeName":"org.kie.workbench.common.forms.editor.client.editor.rendering.EditorFieldLayoutComponent","properties":{"field_id":"field_235","form_id":"f9af0878-6b41-41d9-9307-1754760cd69a"}}]}]},{"height":"12","properties":{},"layoutColumns":[{"span":"12","height":"12","properties":{},"rows":[],"layoutComponents":[{"dragTypeName":"org.kie.workbench.common.forms.editor.client.editor.rendering.EditorFieldLayoutComponent","properties":{"field_id":"field_4012","form_id":"f9af0878-6b41-41d9-9307-1754760cd69a"}}]}]},{"height":"12","properties":{},"layoutColumns":[{"span":"12","height":"12","properties":{},"rows":[],"layoutComponents":[{"dragTypeName":"org.kie.workbench.common.forms.editor.client.editor.rendering.EditorFieldLayoutComponent","properties":{"field_id":"field_4314","form_id":"f9af0878-6b41-41d9-9307-1754760cd69a"}}]}]},{"height":"12","properties":{},"layoutColumns":[{"span":"12","height":"12","properties":{},"rows":[],"layoutComponents":[{"dragTypeName":"org.kie.workbench.common.forms.editor.client.editor.rendering.EditorFieldLayoutComponent","properties":{"field_id":"field_657","form_id":"f9af0878-6b41-41d9-9307-1754760cd69a"}}]}]},{"height":"12","properties":{},"layoutColumns":[{"span":"12","height":"12","properties":{},"rows":[],"layoutComponents":[{"dragTypeName":"org.kie.workbench.common.forms.editor.client.editor.rendering.EditorFieldLayoutComponent","properties":{"field_id":"field_7139","form_id":"f9af0878-6b41-41d9-9307-1754760cd69a"}}]}]},{"height":"12","properties":{},"layoutColumns":[{"span":"12","height":"12","properties":{},"rows":[],"layoutComponents":[{"dragTypeName":"org.kie.workbench.common.forms.editor.client.editor.rendering.EditorFieldLayoutComponent","properties":{"field_id":"field_9469","form_id":"f9af0878-6b41-41d9-9307-1754760cd69a"}}]}]}]}}
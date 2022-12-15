Monthly Report by Plan Package

Monthly_Report_By_Plan_YYYY_MM.zip
The Monthly_Report_By_Plan_YYYY_MM.zip file is a compressed file (using SecureZIP) 
consisting of the following files:

Read_Me_Monthly_Report_By_Plan.txt - this file
Monthly_Report_By_Plan_YYYY_MM.xls - data in Microsoft Excel format
Monthly_Report_By_Plan_YYYY_MM.csv - data in comma separated value (csv) format

Please note that anywhere you see YYYY it denotes the 4 digit year and MM = the 2 digit
month the data covers.

The two data files contain the same base rows of data, but the Excel version includes
header, footer and some totals rows that are not possible in the text version. 


The CSV file contains the following columns (these column names are included as the first row 
of data):

  Contract Number - [text] - The contract identifier 
  Plan ID - [text] - The plan identifier 
  Organization Type - [text] - The type of contract held by the organization with CMS 
  Plan Type  - [text] - The type of plan offered to beneficiaries
  Offers Part D - "Yes" = offers Part D benefit, "No" = does not offer Part D benefit
  Organization Name - [text] - The name of the organization
  Organization Marketing Name - [text] - The name of the organization markets to beneficiaries
  Plan Name - [text] - The name of the plan created by the organization
  Parent Organization - [text] - the name of the parent organization for this contract
  Contract Effective Date - [date] - the data the contract began with CMS (in mm/dd/yyyy h:mm format)
  Enrollment - [text] - total number of beneficiaries enrolled in the plan (* = 10 or less)

This file contains data for the following organization types (where there are active contracts):

  Local CCP
  Regional CCP
  MSA 
  PFFS 
  RFB - PFFS
  Demonstrations 
  National PACE 
  1876 Cost 
  HCPP - 1833 Cost
  Employer/Union Only Direct Contract PDP
  PDP

Special Notes:

  (1) The privacy laws of HIPAA have been interpreted to prohibit publishing 
  enrollment data with values of 10 or less. Data rows with enrollment values
  of 10 or less have been removed from this file. The complete file that includes
  these removed rows but with blanked out enrollment data is also available for
  download. 

  (2) Pilot contracts are excluded from this file. The aggregate enrollment for these contracts
  is available in the Monthly Contract and Enrollment Summary Report.
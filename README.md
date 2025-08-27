# PaperTrail

A local LLM based, secure, document tabulating and organizing system using SQLite, Ollama, and Python


Pipeline:
    Running "Processing Daemon" and Sorting:
        Find file,
        Backup file to target backup location,
        Extract text from OCRs,
        Send to multiple local LLMs,
        Retrieve data from LLMs,
        Achieve consensus and confidence score between responses,
        Store final findings for attributes in temp file for this document,
        Queue up approval for this document,
        Move onto next document.
    "Approval Daemon" Session:
        Store user information for approver tracking,
        Start web server and direct user to website
        Startup queue for
            (A) all processed documents (All previously rejected documents first) OR
            (B) secondary checks,
        If document is approved,
            If first check, put it into second check queue
            If second check, perform
                tabulation (store in database),
                processing (name file and move to target path), and
                cleanup (delete temp files and store logs)
        If document is rejected,
            put it into the rejection folder with the notes on what was done wrong, and
            auto invoke "Processing Daemon" with rejection folder as the folder path.

Flags:
    Invoke "Processing Daemon":
        Give a folder path,
        Recursive search,
        Exclusion of documents,
        Target Path for tabulated documents,
        Location for logs,
        Location for runtime files,
        Keep runtime files,
        Backup files first to another directory incase of failure.
    Running "Processing Daemon" - Track:
        Session duration (file type breakdown and total),
        Files searched (file type breakdown),
        Current file (duration taken for current file),
        Previous file (duration taken for current file),
        Next file (name, location, file type),
        Last 3 timestamps/logs.
    Invoke "Approval Daemon":
        User full name,
        Age,
        Gender.


===============================================================================================================================================================================

Processing Daemon Operations and Document Sorting Workflow:
    Document Discovery and Initial Processing:
        Recursively scan source directory with configurable depth limits and file type filtering,
        Generate SHA-256 checksum for file integrity verification before any processing begins,
        Create unique processing session identifier for batch tracking and audit purposes,
        Validate file accessibility and basic file structure integrity checks,
        Log discovered file with metadata including size, permissions, last modified date, file magic number verification.

    File Protection and Backup Operations:
        Create timestamped backup in designated backup directory structure organized by date and session,
        Preserve original file permissions and metadata during backup process,
        Verify backup integrity by comparing checksums between original and backup copies,
        Establish file locking mechanisms to prevent concurrent access during processing,
        Generate backup manifest file containing original location, backup location, checksums, and timestamps.

    Multi-Engine OCR Text Extraction Process:
        Initialize all configured OCR engines with optimal settings for document type detection,
        Execute Tesseract OCR with language detection and confidence scoring enabled,
        Execute EasyOCR with multi-language support and bounding box detection for layout preservation,
        Execute PaddleOCR with rotation correction and advanced character recognition capabilities,
        Capture per-engine processing time, confidence scores, detected languages, and error conditions,
        Store raw OCR outputs in structured format with engine metadata and processing statistics.

    Multi-Model LLM Processing and Data Extraction:
        Establish fresh context window for primary LLM model with clear system instructions,
        Send Batch 1 prompt containing combined OCR results for basic document identification including title, document type, and primary language detection,
        Refresh LLM context and send Batch 2 prompt for temporal data extraction including creation dates, issue dates, expiration dates, and reception timestamps,
        Refresh LLM context and send Batch 3 prompt for entity identification including issuer names, translator information, officiating bodies, and responsible parties,
        Refresh LLM context and send Batch 4 prompt for classification and metadata including tags, confidentiality levels, utility notes, and additional contextual information,
        Implement retry logic with exponential backoff for failed LLM requests and automatic fallback to secondary models,
        Capture response times, token usage, model performance metrics, and any detected hallucination indicators.

    Consensus Building and Confidence Assessment:
        Compare OCR engine outputs using string similarity algorithms and character-level diff analysis,
        Calculate inter-engine agreement scores and identify conflicting text regions for manual review flagging,
        Analyze LLM batch responses for internal consistency and cross-batch field validation,
        Generate field-level confidence scores based on OCR agreement, LLM consistency, and business rule compliance,
        Create overall document confidence rating weighted by critical field importance and extraction reliability,
        Flag documents requiring manual review based on configurable confidence thresholds and disagreement patterns.

    Temporary Storage and Queue Management:
        Generate document processing session folder structure with unique session identifiers,
        Store intermediate processing results in JSON format with full audit trail and processing metadata,
        Create approval queue entry with document reference, confidence scores, review priorities, and processing history,
        Preserve all raw OCR outputs, LLM responses, and processing logs for debugging and quality improvement,
        Update global processing statistics and progress tracking databases with current document status.

    Workflow Continuation and Resource Management:
        Release file locks and cleanup temporary processing resources,
        Update processing daemon status and move to next document in queue,
        Implement resource monitoring for memory usage, GPU utilization, and processing efficiency,
        Generate processing session summary with statistics, error counts, and performance metrics.

Approval Daemon Session Management and Review Process:
    User Authentication and Session Initialization:
        Capture approver identification including full name, employee identifier, role, and security clearance level,
        Record session start time, IP address, browser information, and authentication method for audit compliance,
        Initialize approval session database with user tracking and decision history for accountability,
        Configure user-specific approval preferences including display settings, review priorities, and notification preferences.

    Web Server Initialization and Interface Setup:
        Launch Streamlit or Flask web server on configurable port with SSL/TLS encryption for secure access,
        Load approval queue management system with real-time status updates and collaborative review capabilities,
        Initialize document preview system supporting multiple file formats with zoom, rotation, and annotation features,
        Setup approval workflow tracking with decision logging, timestamp recording, and user action history.

    Queue Management and Document Prioritization:
        Primary Queue Loading for First-Time Reviews:
            Load all previously rejected documents first sorted by rejection timestamp and priority level,
            Load new processed documents sorted by confidence score with lowest confidence items prioritized for review,
            Display queue statistics including total pending documents, average processing time, and estimated completion timeline,
            Implement queue filtering by document type, confidence level, processing date, and custom tag criteria.

        Secondary Queue Loading for Final Approval Reviews:
            Load all first-approved documents sorted by first approval timestamp and business criticality,
            Display first approver information, approval timestamp, and any notes or conditions attached to initial approval,
            Show confidence score changes since first approval and any updated processing information,
            Implement secondary reviewer assignment logic to ensure different person reviews each document for approval separation.

    Document Approval Workflow Processing:
        First Approval Decision Processing:
            Display document preview with full metadata extraction results and confidence scoring breakdown,
            Show OCR cross-validation results with engine agreement analysis and any conflicting text regions,
            Present LLM extraction confidence scores with field-by-field reliability indicators and uncertainty flags,
            Enable inline editing of individual metadata fields with automatic reprocessing triggers and validation checks,
            Capture approval decision with detailed reasoning, approval conditions, and any required follow-up actions,
            Move approved documents to secondary approval queue with first approver metadata and decision history,
            Generate approval notification for secondary reviewer assignment and timeline establishment.

        Secondary Approval Decision Processing:
            Display first approval details including original approver, decision reasoning, and any modification history,
            Show side-by-side comparison of original extraction versus any modifications made during first approval,
            Enable final metadata review with business rule validation and compliance checking capabilities,
            Capture final approval decision with secondary approver authentication and decision justification,
            Trigger final document tabulation process including database storage, file organization, and archival procedures.

        Document Tabulation and Final Processing:
            Insert document record into SQLite database with all approved metadata and processing history,
            Generate unique password for document encryption using cryptographically secure random generation,
            Encrypt original document file using approved password and AES-256 encryption with authenticated modes,
            Rename and move encrypted file to target directory structure organized by document type, date, and classification level,
            Update document tracking database with final file locations, encryption status, and access control information,
            Generate processing completion report with statistics, quality metrics, and audit trail information.

        Cleanup and Session Management:
            Delete temporary processing files and intermediate data storage with secure deletion methods,
            Archive processing logs to permanent storage with compression and retention policy compliance,
            Update approval session statistics with decision counts, processing times, and efficiency metrics,
            Generate session completion report with approver activity summary and quality control statistics.

    Document Rejection Handling and Reprocessing:
        Rejection Documentation and Analysis:
            Capture detailed rejection reasoning with specific field-level feedback and improvement suggestions,
            Store rejection notes in structured format with category classification and priority assignment,
            Move rejected document to rejection folder with organized subfolder structure by rejection type and date,
            Generate reprocessing instruction file containing human feedback and specific processing guidance for daemon restart.

        Automatic Reprocessing Trigger:
            Invoke Processing Daemon with rejection folder as source directory and enhanced processing parameters,
            Include human feedback in LLM prompts for improved extraction accuracy and field-specific guidance,
            Apply rejection-specific processing logic including additional OCR engines or alternative LLM models,
            Implement enhanced validation rules based on specific rejection feedback and common error patterns.

===============================================================================================================================================================================

Processing Daemon Invocation Arguments:
    Source Directory Configuration:
        source_folder_path: Absolute path to directory containing documents for processing with read permission validation,
        recursive_search_enabled: Boolean flag enabling subdirectory traversal with configurable depth limits,
        max_recursion_depth: Integer limit for directory traversal depth to prevent infinite recursion,
        follow_symlinks: Boolean flag for symbolic link following with circular reference detection.

    Document Filtering and Selection:
        included_file_extensions: List of allowed file extensions with case-insensitive matching,
        excluded_file_patterns: Regular expression patterns for file exclusion with glob pattern support,
        exclude_previously_processed: Skip files already present in database based on checksum comparison.

    Target Location and Organization:
        target_directory_for_processed_documents: Destination path for approved and encrypted documents with automatic subdirectory creation,
        directory_organization_scheme: Folder structure pattern using date, document type, and classification variables,
        filename_naming_convention: Template for processed file naming using UUID, original name, and metadata fields,
        backup_location_for_originals: Separate backup directory for original unprocessed files with timestamp organization.

    Logging and Runtime Configuration:
        log_output_directory: Path for processing logs with automatic rotation and archival policies,
        log_detail_level: Configurable verbosity from minimal to comprehensive debugging information,
        runtime_files_directory: Temporary storage for intermediate processing files and session data,
        keep_runtime_files_after_completion: Boolean flag for debugging and audit trail preservation,
        processing_session_timeout: Maximum time allowed for single document processing before automatic failure.

    Performance and Resource Management:
        concurrent_processing_threads: Number of parallel document processing pipelines for efficiency optimization,
        gpu_memory_limit: VRAM allocation limit for LLM processing with automatic fallback to CPU processing,
        ocr_engine_timeout_seconds: Maximum time allowed per OCR engine before declaring failure and continuing,
        llm_response_timeout_seconds: Maximum wait time for LLM responses with automatic retry logic,
        batch_size_for_llm_processing: Number of documents processed before LLM context refresh.

Processing Daemon Real-Time Monitoring and Statistics:
    Session Performance Tracking:
        total_session_duration_seconds: Elapsed time since processing daemon startup with efficiency calculations,
        documents_processed_count: Total number of documents completed with success and failure breakdowns,
        current_processing_rate_per_hour: Real-time calculation of documents processed per hour with trending analysis,
        estimated_completion_time: Projected finish time based on current processing rate and remaining queue size,
        resource_utilization_metrics: CPU usage, GPU utilization, memory consumption, and disk I/O statistics.

    File Type Analysis and Breakdown:
        files_discovered_by_type: Dictionary containing counts for each file extension with size statistics,
        processing_time_by_file_type: Average processing duration for different document types with variance analysis,
        success_rate_by_file_type: Percentage of successful extractions by document format with error pattern analysis,
        confidence_scores_by_type: Average extraction confidence broken down by document type and complexity.

    Current Document Processing Status:
        current_file_being_processed: Full path, filename, size, and estimated completion time for active document,
        current_processing_stage: Detailed status including OCR engine progress, LLM batch completion, and validation stage,
        current_document_duration: Time elapsed for current document with stage-by-stage breakdown,
        current_document_confidence: Real-time confidence scoring as each processing stage completes.

    Historical Processing Information:
        previous_document_summary: Complete processing report for last completed document with success metrics,
        previous_document_processing_duration: Total time taken for previous document with stage timing breakdown,
        last_three_completed_documents: Queue history with processing times, confidence scores, and final disposition,
        next_document_preview: Filename, estimated processing time, and preliminary file analysis for upcoming document.

    Real-Time Logging and Status Updates:
        last_three_system_timestamps: Recent log entries with severity levels and processing stage indicators,
        error_count_current_session: Total errors encountered with categorization by error type and severity,
        warning_count_current_session: Non-fatal issues detected with resolution status and impact assessment,
        critical_alerts_active: System-level issues requiring immediate attention with escalation procedures.

Approval Daemon Invocation Parameters:
    Approver Identity and Authentication:
        approver_full_name: Complete legal name for audit trail and approval authority verification,
        employee_identification_number: Unique identifier for approval authority validation and access control,
        security_clearance_level: Clearance level for confidential document access and approval authority limits,
        approval_authority_scope: Document types and classification levels this approver is authorized to review.

    Session Configuration and Security:
        session_timeout_minutes: Automatic logout time for security compliance and idle session management,
        require_two_factor_authentication: Boolean flag for enhanced security using hardware tokens or mobile authentication,
        audit_trail_detail_level: Comprehensive logging level for compliance and security audit requirements,
        approval_delegation_allowed: Permission for approver to delegate reviews to other authorized personnel.

    Review Interface and Workflow Settings:
        documents_per_page_display: Number of documents shown simultaneously for efficiency without overwhelming interface,
        default_preview_zoom_level: Initial document preview magnification for optimal readability,
        auto_advance_after_approval: Automatic progression to next document versus manual navigation control,
        rejection_note_templates: Pre-configured rejection reason templates for consistency and efficiency.

    Quality Control and Validation Parameters:
        minimum_confidence_threshold_for_approval: Automatic rejection threshold for documents below specified confidence levels,
        require_justification_for_low_confidence_approval: Mandatory explanation when approving documents below confidence threshold,
        mandatory_secondary_review_for_types: Document types requiring dual approval regardless of confidence score,
        escalation_rules_for_complex_documents: Automatic routing to senior approvers based on document complexity or classification.

===============================================================================================================================================================================
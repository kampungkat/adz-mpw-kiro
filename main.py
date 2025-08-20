"""
Main entry point for the AI Media Planner application.
"""
import os
import streamlit as st
from config.settings import config_manager
from data.manager import DataManager
from ui.components import MediaPlannerForm, FormatSelectionComponent, display_validation_summary

# Add this at the top of main.py for debugging
st.write("**Debug: Files in root directory:**")
root_files = os.listdir(".")
st.write(root_files)

st.write("**Debug: Detailed file validation:**")

try:
    data_manager = DataManager()
    validation_result = data_manager.validate_data_freshness()
    
    st.write("Validation result:", validation_result)
    
    # Try to load each file individually
    st.write("**Testing rate card loading:**")
    try:
        rate_data = data_manager.load_rate_cards()
        st.success(f"✅ Rate card loaded successfully! Found {len(rate_data.get('markets', []))} markets")
    except Exception as e:
        st.error(f"❌ Rate card loading failed: {str(e)}")
    
    st.write("**Testing site list loading:**")
    try:
        site_data = data_manager.load_site_lists()
        st.success(f"✅ Site list loaded successfully! Found {len(site_data.get('markets', {}))} markets")
    except Exception as e:
        st.error(f"❌ Site list loading failed: {str(e)}")
        
except Exception as e:
    st.error(f"DataManager initialization failed: {str(e)}")


# Check specifically for your files
rate_card_exists = os.path.exists("adzymic_rate_card.xlsx")
site_list_exists = os.path.exists("APX Sitelist - Regional.xlsx")

st.write(f"Rate card exists: {rate_card_exists}")
st.write(f"Site list exists: {site_list_exists}")

def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="AI Media Planner",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🎯 AI Media Planner")
    st.markdown("Generate intelligent media plans for your advertising campaigns")
    
    # Load configuration
    try:
        config = config_manager.load_config()
        st.success("✅ Configuration loaded successfully")
    except ValueError as e:
        st.error(f"❌ Configuration Error: {e}")
        st.stop()
    
    # Initialize data manager with comprehensive error handling
    try:
        data_manager = DataManager(cache_ttl_hours=config.cache_timeout_hours)
        
        # Try to actually load the data instead of just checking cache
        data_load_success = True
        load_errors = []
        
        try:
            # Attempt to load rate card data
            rate_data = data_manager.load_rate_cards()
            st.success(f"✅ Rate card loaded successfully! Found {len(rate_data.get('markets', []))} markets")
        except Exception as e:
            data_load_success = False
            load_errors.append(f"Rate card loading failed: {str(e)}")
        
        try:
            # Attempt to load site list data
            site_data = data_manager.load_site_lists()
            st.success(f"✅ Site list loaded successfully! Found {len(site_data.get('markets', {}))} markets")
        except Exception as e:
            data_load_success = False
            load_errors.append(f"Site list loading failed: {str(e)}")
        
        # If data loading failed, show detailed error information
        if not data_load_success:
            st.error("❌ **Data Loading Failed**")
            st.error("The application could not load the required data files.")
            
            with st.expander("📋 Error Details and Setup Instructions"):
                st.write("**Loading Errors:**")
                for error in load_errors:
                    st.write(f"• {error}")
                st.write("")
                st.write("**Required Files:**")
                st.write("• `adzymic_rate_card.xlsx` - Contains CPM pricing data for different markets and ad formats")
                st.write("• `APX Sitelist - Regional.xlsx` - Contains site categorization and availability by market")
                st.write("")
                st.write("**Setup Instructions:**")
                st.write("1. Ensure files are in Excel format (.xlsx or .xls)")
                st.write("2. Check that files are not password protected")
                st.write("3. Verify file structure matches expected format")
                st.write("4. Try re-uploading files if they appear corrupted")
                st.write("")
                st.write("**File Format Requirements:**")
                st.write("• Rate card: Market codes as columns, ad format names as rows, CPM values in cells")
                st.write("• Site list: Separate sheets for each market with site categorization")
            
            st.stop()
            
        elif validation_result['overall_status'] == 'needs_refresh':
            st.warning("⚠️ **Data May Be Outdated**")
            st.warning("Some data files may be stale. Consider refreshing for best results.")
            
            # Show specific recommendations
            recommendations = []
            if validation_result['rate_card'].get('recommendations'):
                recommendations.extend(validation_result['rate_card']['recommendations'])
            if validation_result['site_list'].get('recommendations'):
                recommendations.extend(validation_result['site_list']['recommendations'])
            
            if recommendations:
                with st.expander("💡 Data Refresh Recommendations"):
                    for rec in recommendations:
                        st.write(f"• {rec}")
        
        elif validation_result['overall_status'] == 'error':
            st.error("❌ **Data Validation Error**")
            st.error("An error occurred while validating data files. Please check file integrity and permissions.")
            
            with st.expander("🔧 Troubleshooting Data Errors"):
                st.write("**Common Issues:**")
                st.write("• File corruption: Re-download data files")
                st.write("• Permission issues: Check file read permissions")
                st.write("• Format issues: Ensure files are valid Excel format")
                st.write("• Network issues: Check file accessibility")
            
            st.stop()
        
    except FileNotFoundError as e:
        st.error("❌ **Data Files Not Found**")
        st.error(f"Cannot locate required data files: {str(e)}")
        st.info("Please ensure 'adzymic_rate_card.xlsx' and 'APX Sitelist - Regional.xlsx' are in the application directory.")
        st.stop()
        
    except PermissionError as e:
        st.error("❌ **File Permission Error**")
        st.error(f"Cannot access data files due to permission restrictions: {str(e)}")
        st.info("Please check file permissions and ensure the application has read access to data files.")
        st.stop()
        
    except Exception as e:
        st.error("❌ **Data Manager Initialization Error**")
        st.error(f"Unexpected error initializing data management system: {str(e)}")
        
        with st.expander("🔧 Error Details and Support"):
            st.write("**Error Information:**")
            st.code(str(e))
            st.write("")
            st.write("**Next Steps:**")
            st.write("1. Check application logs for detailed error information")
            st.write("2. Verify all data files are present and accessible")
            st.write("3. Restart the application")
            st.write("4. Contact technical support if the issue persists")
        
        st.stop()
    
    # Initialize UI components
    form_component = MediaPlannerForm(data_manager)
    format_component = FormatSelectionComponent(data_manager)
    
    # Render main form
    form_data, form_submitted = form_component.render()
    
    # Handle manual format selection
    format_data = {}
    if form_data.get('planning_mode') == 'Manual Selection' and form_data.get('country') and form_data.get('budget', 0) > 0:
        format_data = format_component.render(form_data['country'], form_data['budget'])
    
    # Display summary and next steps
    if form_submitted:
        display_validation_summary(form_data, format_data)
        
        # Check if ready for plan generation
        ready_for_generation = True
        
        if form_data.get('planning_mode') == 'Manual Selection':
            if not format_data.get('selected_formats'):
                st.warning("⚠️ Please select at least one ad format for manual planning mode.")
                ready_for_generation = False
            elif format_data.get('budget_utilization', 0) > 110:
                st.error("❌ Budget allocation exceeds available budget. Please adjust format allocations.")
                ready_for_generation = False
        
        if ready_for_generation:
            st.success("✅ Ready for media plan generation!")
            st.info("🚧 Plan generation will be implemented in subsequent tasks.")
            
            # Store validated data in session state for next steps
            st.session_state['validated_form_data'] = form_data
            st.session_state['validated_format_data'] = format_data
    
    # Display current configuration (for development)
    with st.expander("System Information"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Configuration:**")
            st.write(f"Max Plans: {config.max_plans_generated}")
            st.write(f"Default Currency: {config.default_currency}")
            st.write(f"Cache Timeout: {config.cache_timeout_hours} hours")
        
        with col2:
            st.write("**Data Status:**")
            st.write(f"Overall Status: {validation_result['overall_status']}")
            st.write(f"Rate Card: {validation_result['rate_card']['status']}")
            st.write(f"Site List: {validation_result['site_list']['status']}")
        
        # Available markets
        try:
            markets = data_manager.get_available_markets()
            st.write(f"**Available Markets:** {', '.join(markets) if markets else 'None'}")
        except Exception:
            st.write("**Available Markets:** Error loading")


if __name__ == "__main__":
    main()
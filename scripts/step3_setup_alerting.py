"""
Step 3: Set up Alerting and Notifications
Configure comprehensive alerting rules with SMS, email, and webhook notifications
"""

import json
from pathlib import Path
from datetime import datetime

def print_alerting_setup():
    """Print alerting and notification setup guide"""
    
    print("🚨 STEP 3: SET UP ALERTING AND NOTIFICATIONS")
    print("=" * 60)
    print("Account: yyacoup")
    print("Region: AF-Cairo (af-north-1)")
    print("Scope: Critical, Warning, and Business alerts")
    print()
    
    print("🌐 1. ACCESS NOTIFICATION SETUP:")
    print("   URL: https://console.huaweicloud.com/smn")
    print("   Service: Simple Message Notification (SMN)")
    print("   Navigate: SMN → Topics")
    print()
    
    print("📱 2. CREATE NOTIFICATION TOPICS:")
    topics = [
        {"name": "ARSL-Critical-Alerts", "description": "Critical system alerts requiring immediate attention"},
        {"name": "ARSL-Warning-Alerts", "description": "Warning alerts for monitoring and investigation"},
        {"name": "ARSL-Business-Alerts", "description": "Business metric alerts and daily reports"}
    ]
    
    for topic in topics:
        print(f"   📢 {topic['name']}")
        print(f"     Description: {topic['description']}")
        print()

def create_notification_topics():
    """Create notification topic configurations"""
    
    topics_config = {
        "notification_topics": [
            {
                "topic_name": "ARSL-Critical-Alerts",
                "display_name": "ARSL Critical System Alerts",
                "description": "Critical alerts requiring immediate response",
                "subscription_types": ["SMS", "Email", "HTTP/HTTPS"],
                "subscribers": [
                    {
                        "protocol": "sms",
                        "endpoint": "+201234567890",  # Replace with actual phone number
                        "filter_policy": {
                            "severity": ["critical", "emergency"]
                        }
                    },
                    {
                        "protocol": "email",
                        "endpoint": "alerts@yourdomain.com",  # Replace with actual email
                        "filter_policy": {
                            "severity": ["critical", "emergency", "warning"]
                        }
                    },
                    {
                        "protocol": "http",
                        "endpoint": "https://your-webhook-url.com/alerts",
                        "filter_policy": {
                            "severity": ["critical", "emergency"]
                        }
                    }
                ]
            },
            
            {
                "topic_name": "ARSL-Warning-Alerts", 
                "display_name": "ARSL Warning Alerts",
                "description": "Performance and operational warnings",
                "subscription_types": ["Email", "HTTP/HTTPS"],
                "subscribers": [
                    {
                        "protocol": "email",
                        "endpoint": "monitoring@yourdomain.com",
                        "filter_policy": {
                            "severity": ["warning", "info"]
                        }
                    },
                    {
                        "protocol": "http",
                        "endpoint": "https://your-monitoring-webhook.com/warnings",
                        "filter_policy": {
                            "component": ["api", "model", "system"]
                        }
                    }
                ]
            },
            
            {
                "topic_name": "ARSL-Business-Alerts",
                "display_name": "ARSL Business Metrics",
                "description": "Business analytics and daily reports",
                "subscription_types": ["Email"],
                "subscribers": [
                    {
                        "protocol": "email", 
                        "endpoint": "business@yourdomain.com",
                        "filter_policy": {
                            "report_type": ["daily", "weekly", "monthly"]
                        }
                    }
                ]
            }
        ]
    }
    
    # Save topics configuration
    topics_file = Path("config/notification_topics.json")
    topics_file.parent.mkdir(exist_ok=True)
    
    with open(topics_file, 'w') as f:
        json.dump(topics_config, f, indent=2)
    
    print(f"📱 Notification topics saved: {topics_file}")
    return topics_file

def create_alerting_rules():
    """Create comprehensive alerting rules"""
    
    alerting_rules = {
        "critical_alerts": [
            {
                "alert_name": "ECS_Instance_Down",
                "description": "ECS instance is not responding or has stopped",
                "metric": {
                    "namespace": "SYS.ECS",
                    "metric_name": "vm_status",
                    "dimensions": {"instance_id": "i-xxxxxxxxx"},
                    "statistic": "Maximum"
                },
                "condition": {
                    "comparison_operator": "LessThanThreshold",
                    "threshold": 1,
                    "evaluation_periods": 2,
                    "period": 60
                },
                "alarm_actions": {
                    "notification_topic": "ARSL-Critical-Alerts",
                    "auto_scaling": False,
                    "webhook": True
                },
                "severity": "critical",
                "estimated_recovery_time": "5-15 minutes"
            },
            
            {
                "alert_name": "API_Service_Unavailable",
                "description": "API Gateway or inference service is down",
                "metric": {
                    "namespace": "ARSL/API",
                    "metric_name": "health_check_response_time",
                    "statistic": "Average"
                },
                "condition": {
                    "comparison_operator": "LessThanThreshold",
                    "threshold": 0.1,  # No response
                    "evaluation_periods": 3,
                    "period": 60
                },
                "alarm_actions": {
                    "notification_topic": "ARSL-Critical-Alerts",
                    "auto_scaling": True,
                    "webhook": True
                },
                "severity": "critical",
                "estimated_recovery_time": "2-10 minutes"
            },
            
            {
                "alert_name": "High_Error_Rate",
                "description": "API error rate exceeds acceptable threshold",
                "metric": {
                    "namespace": "SYS.APIG",
                    "metric_name": "error_rate",
                    "statistic": "Average"
                },
                "condition": {
                    "comparison_operator": "GreaterThanThreshold",
                    "threshold": 10,  # 10% error rate
                    "evaluation_periods": 2,
                    "period": 300
                },
                "alarm_actions": {
                    "notification_topic": "ARSL-Critical-Alerts",
                    "auto_scaling": False,
                    "webhook": True
                },
                "severity": "critical",
                "estimated_recovery_time": "10-30 minutes"
            },
            
            {
                "alert_name": "Resource_Exhaustion",
                "description": "System resources (CPU/Memory) critically high",
                "metric": {
                    "namespace": "SYS.ECS",
                    "metric_name": "cpu_util",
                    "statistic": "Average"
                },
                "condition": {
                    "comparison_operator": "GreaterThanThreshold", 
                    "threshold": 90,  # 90% CPU
                    "evaluation_periods": 3,
                    "period": 300
                },
                "alarm_actions": {
                    "notification_topic": "ARSL-Critical-Alerts",
                    "auto_scaling": True,
                    "webhook": True
                },
                "severity": "critical",
                "estimated_recovery_time": "5-20 minutes"
            }
        ],
        
        "warning_alerts": [
            {
                "alert_name": "High_Response_Time",
                "description": "API response time degradation",
                "metric": {
                    "namespace": "SYS.APIG",
                    "metric_name": "latency",
                    "statistic": "Average"
                },
                "condition": {
                    "comparison_operator": "GreaterThanThreshold",
                    "threshold": 1000,  # 1000ms
                    "evaluation_periods": 3,
                    "period": 600
                },
                "alarm_actions": {
                    "notification_topic": "ARSL-Warning-Alerts",
                    "auto_scaling": False,
                    "webhook": True
                },
                "severity": "warning",
                "estimated_recovery_time": "15-45 minutes"
            },
            
            {
                "alert_name": "Memory_Usage_High",
                "description": "Memory utilization approaching limits",
                "metric": {
                    "namespace": "SYS.ECS",
                    "metric_name": "mem_util",
                    "statistic": "Average"
                },
                "condition": {
                    "comparison_operator": "GreaterThanThreshold",
                    "threshold": 75,  # 75% memory
                    "evaluation_periods": 4,
                    "period": 300
                },
                "alarm_actions": {
                    "notification_topic": "ARSL-Warning-Alerts",
                    "auto_scaling": False,
                    "webhook": False
                },
                "severity": "warning",
                "estimated_recovery_time": "30-60 minutes"
            },
            
            {
                "alert_name": "Storage_Usage_High",
                "description": "OBS storage usage approaching quota",
                "metric": {
                    "namespace": "SYS.OBS",
                    "metric_name": "storage_size",
                    "statistic": "Maximum"
                },
                "condition": {
                    "comparison_operator": "GreaterThanThreshold",
                    "threshold": 1000000,  # 1TB (adjust based on quota)
                    "evaluation_periods": 1,
                    "period": 3600  # Check hourly
                },
                "alarm_actions": {
                    "notification_topic": "ARSL-Warning-Alerts",
                    "auto_scaling": False,
                    "webhook": False
                },
                "severity": "warning",
                "estimated_recovery_time": "1-4 hours"
            }
        ],
        
        "business_alerts": [
            {
                "alert_name": "Model_Accuracy_Drop",
                "description": "Model prediction accuracy below acceptable level",
                "metric": {
                    "namespace": "ARSL/Business",
                    "metric_name": "prediction_accuracy",
                    "statistic": "Average"
                },
                "condition": {
                    "comparison_operator": "LessThanThreshold",
                    "threshold": 85,  # 85% accuracy
                    "evaluation_periods": 6,
                    "period": 600  # 10-minute periods
                },
                "alarm_actions": {
                    "notification_topic": "ARSL-Business-Alerts",
                    "auto_scaling": False,
                    "webhook": True
                },
                "severity": "warning",
                "estimated_recovery_time": "2-8 hours"
            },
            
            {
                "alert_name": "Unusual_Traffic_Pattern",
                "description": "Unexpected increase in API traffic",
                "metric": {
                    "namespace": "SYS.APIG",
                    "metric_name": "req_count",
                    "statistic": "Sum"
                },
                "condition": {
                    "comparison_operator": "GreaterThanThreshold",
                    "threshold": 1000,  # 1000 requests in period
                    "evaluation_periods": 2,
                    "period": 600
                },
                "alarm_actions": {
                    "notification_topic": "ARSL-Business-Alerts",
                    "auto_scaling": True,
                    "webhook": True
                },
                "severity": "info",
                "estimated_recovery_time": "Monitor and analyze"
            }
        ]
    }
    
    # Save alerting rules
    rules_file = Path("config/alerting_rules.json")
    rules_file.parent.mkdir(exist_ok=True)
    
    with open(rules_file, 'w') as f:
        json.dump(alerting_rules, f, indent=2)
    
    print(f"🚨 Alerting rules saved: {rules_file}")
    return rules_file

def create_alert_response_playbook():
    """Create incident response playbook"""
    
    playbook = {
        "incident_response_playbook": {
            "critical_alerts": {
                "ECS_Instance_Down": {
                    "immediate_actions": [
                        "Check ECS console for instance status",
                        "Attempt to restart instance if stopped",
                        "Check security group and network settings",
                        "Review system logs for crash indicators",
                        "If unable to restart, launch new instance"
                    ],
                    "escalation_timeline": "15 minutes",
                    "recovery_sla": "30 minutes",
                    "rollback_plan": "Launch backup instance, redirect traffic"
                },
                
                "API_Service_Unavailable": {
                    "immediate_actions": [
                        "Check API Gateway console status",
                        "Verify ModelArts inference service status",
                        "Test direct service endpoints",
                        "Check service logs for errors",
                        "Restart services if necessary"
                    ],
                    "escalation_timeline": "10 minutes",
                    "recovery_sla": "20 minutes",
                    "rollback_plan": "Switch to backup API endpoint"
                },
                
                "High_Error_Rate": {
                    "immediate_actions": [
                        "Identify error patterns in logs",
                        "Check model service health",
                        "Verify input data quality",
                        "Review recent deployments",
                        "Implement circuit breaker if needed"
                    ],
                    "escalation_timeline": "20 minutes",
                    "recovery_sla": "45 minutes",
                    "rollback_plan": "Revert to previous model version"
                },
                
                "Resource_Exhaustion": {
                    "immediate_actions": [
                        "Identify resource-consuming processes",
                        "Scale up instance if auto-scaling failed",
                        "Implement rate limiting",
                        "Clear temporary files and caches",
                        "Restart services to free memory"
                    ],
                    "escalation_timeline": "15 minutes",
                    "recovery_sla": "30 minutes",
                    "rollback_plan": "Reduce traffic load, scale horizontally"
                }
            },
            
            "warning_alerts": {
                "High_Response_Time": {
                    "investigation_steps": [
                        "Check database performance",
                        "Review model inference time",
                        "Analyze network latency",
                        "Check for resource contention",
                        "Review recent code changes"
                    ],
                    "monitoring_period": "2 hours",
                    "escalation_threshold": "No improvement in 2 hours"
                },
                
                "Memory_Usage_High": {
                    "investigation_steps": [
                        "Identify memory-consuming processes",
                        "Check for memory leaks",
                        "Review application logs",
                        "Analyze garbage collection patterns",
                        "Consider memory optimization"
                    ],
                    "monitoring_period": "4 hours",
                    "escalation_threshold": "Memory usage >85%"
                }
            },
            
            "contact_information": {
                "on_call_engineer": "+201234567890",
                "escalation_manager": "+201234567891",
                "business_contact": "business@yourdomain.com",
                "technical_lead": "tech@yourdomain.com"
            },
            
            "communication_channels": {
                "emergency": "SMS + Phone call",
                "urgent": "Email + Slack",
                "normal": "Email + Ticket system"
            }
        }
    }
    
    # Save playbook
    playbook_file = Path("config/incident_response_playbook.json")
    playbook_file.parent.mkdir(exist_ok=True)
    
    with open(playbook_file, 'w') as f:
        json.dump(playbook, f, indent=2)
    
    print(f"📖 Incident response playbook saved: {playbook_file}")
    return playbook_file

def print_setup_steps():
    """Print step-by-step alerting setup instructions"""
    
    print("\n📋 ALERTING SETUP STEPS:")
    print("=" * 30)
    
    steps = [
        {
            "step": "1. Create SMN Topics",
            "duration": "10 minutes",
            "actions": [
                "Navigate to SMN console",
                "Create ARSL-Critical-Alerts topic",
                "Create ARSL-Warning-Alerts topic", 
                "Create ARSL-Business-Alerts topic",
                "Configure topic policies and permissions"
            ]
        },
        {
            "step": "2. Add Subscribers",
            "duration": "15 minutes",
            "actions": [
                "Add SMS subscription for critical alerts",
                "Add email subscriptions for all topics",
                "Add webhook endpoints for automation",
                "Configure subscription filters",
                "Test subscription confirmations"
            ]
        },
        {
            "step": "3. Create Alarm Rules",
            "duration": "25 minutes",
            "actions": [
                "Navigate to Cloud Eye → Alarm Rules",
                "Create critical alert rules (4 rules)",
                "Create warning alert rules (3 rules)",
                "Create business alert rules (2 rules)",
                "Configure alarm actions and notifications",
                "Test alarm rule conditions"
            ]
        },
        {
            "step": "4. Configure Auto-scaling",
            "duration": "12 minutes",
            "actions": [
                "Set up auto-scaling policies",
                "Link scaling actions to alert rules",
                "Configure scaling cooldown periods",
                "Test scaling triggers",
                "Verify scaling limits and costs"
            ]
        },
        {
            "step": "5. Set Up Webhook Integration",
            "duration": "15 minutes",
            "actions": [
                "Configure webhook endpoints",
                "Test webhook payloads",
                "Set up webhook authentication",
                "Implement webhook handlers",
                "Test end-to-end webhook flow"
            ]
        },
        {
            "step": "6. Test Alert System",
            "duration": "20 minutes",
            "actions": [
                "Trigger test alerts manually",
                "Verify SMS notifications work",
                "Check email alert formatting",
                "Test webhook delivery",
                "Validate escalation procedures",
                "Document test results"
            ]
        }
    ]
    
    total_time = 0
    for step in steps:
        print(f"\n   🔸 {step['step']} ({step['duration']}):")
        for action in step['actions']:
            print(f"     • {action}")
        total_time += int(step['duration'].split()[0])
    
    print(f"\n⏱️ Total setup time: ~{total_time} minutes")
    print(f"🚨 Alert rules: 9 total (4 critical + 3 warning + 2 business)")

def main():
    """Main function"""
    print("🚨 STEP 3: SET UP ALERTING AND NOTIFICATIONS")
    print("Account: yyacoup")
    print("Region: AF-Cairo (af-north-1)")
    print("Target: Comprehensive alerting system")
    print("=" * 60)
    
    # Print alerting setup guide
    print_alerting_setup()
    
    # Create notification topics
    topics_file = create_notification_topics()
    
    # Create alerting rules
    rules_file = create_alerting_rules()
    
    # Create response playbook
    playbook_file = create_alert_response_playbook()
    
    # Print setup steps
    print_setup_steps()
    
    print(f"\n🎯 STEP 3 SUMMARY:")
    print(f"✅ Notification topics configured ({topics_file})")
    print(f"✅ Alerting rules defined ({rules_file})")
    print(f"✅ Incident response playbook created ({playbook_file})")
    print(f"✅ Setup procedures documented")
    print(f"📋 Ready for alerting system deployment")
    print(f"🌐 Next: Test end-to-end monitoring")
    
    print(f"\n💡 QUICK START:")
    print(f"1. Go to: https://console.huaweicloud.com/smn")
    print(f"2. Create topics using: {topics_file}")
    print(f"3. Set up alerts using: {rules_file}")
    print(f"4. Test notification delivery")
    print(f"5. Proceed to Step 4: End-to-end testing")

if __name__ == "__main__":
    main()